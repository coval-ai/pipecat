#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Coval WebRTC Transport.

This module provides a WebRTC transport for connecting to Coval's cloud agents
by extending Pipecat's SmallWebRTCConnection and SmallWebRTCTransport.
"""

import asyncio
import copy
import json
import time
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import StartFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import (
    SmallWebRTCOutputTransport,
    SmallWebRTCTransport,
)
from pipecat.transports.network.webrtc_connection import (
    SIGNALLING_TYPE,
    RTCSessionDescription,
    SmallWebRTCConnection,
)


class LocalCovalDailyOutputTransport(SmallWebRTCOutputTransport):
    """This output transport for Coval Daily connections overrides the default start
    behavior to prevent redundant connection attempts. The input transport is
    responsible for initiating the WebRTC connection.
    """

    async def start(self, frame: StartFrame):
        logger.debug(f"Starting {self.name} output transport")
        # We only want to call the grandparent's start method to avoid the
        # connection logic in SmallWebRTCOutputTransport.
        await super(SmallWebRTCOutputTransport, self).start(frame)


class LocalCovalDailyConnection(SmallWebRTCConnection):
    """WebRTC connection that integrates Coval's cloud agent signaling.

    This connection class implements a "vanilla ICE" flow, where all ICE
    candidates are gathered before the offer is sent. This is in contrast to
    the base class's "trickle ICE" implementation.
    """

    def __init__(
        self,
        api_key: str,
        signaling_base_url: str,
        session_status_base_url: str,
        launch_evaluation_url: str,
        evaluation_config: Dict[str, Any],
        dev_mode: bool = False,
        client_metadata: Optional[dict] = None,
        ice_servers: Optional[List[str]] = None,
    ):
        # We are the initiator in this flow.
        super().__init__(ice_servers=ice_servers)
        self._initiator = True  # This connection is always the initiator

        self._api_key = api_key
        self._signaling_base_url = signaling_base_url
        self._session_status_base_url = session_status_base_url
        self._launch_evaluation_url = launch_evaluation_url
        self._evaluation_config = evaluation_config
        self._dev_mode = dev_mode
        self._client_metadata = client_metadata or {}

        # Extract required fields for signaling
        self._organization_id = self._evaluation_config.get("organization_id")
        self._agent_id = self._evaluation_config.get("agent_id")

        if not self._organization_id or not self._agent_id:
            raise ValueError("evaluation_config must contain 'organization_id' and 'agent_id'")

        self._session_id: Optional[str] = None
        self._evaluation_run_id: Optional[str] = None
        self._ice_gathering_complete = asyncio.Event()
        self._offer: Optional[RTCSessionDescription] = None
        self._data_channel_name = "data"
        self._connect_lock = asyncio.Lock()

    async def create_offer(self) -> Dict[str, Any]:
        """Creates an SDP offer and waits for all ICE candidates to be gathered."""
        if not self._initiator:
            raise RuntimeError("Attempted to create an offer while not in initiator mode")

        # Setup data channel and transceivers
        self._data_channel = self._pc.createDataChannel(self._data_channel_name)

        @self._data_channel.on("open")
        async def on_open():
            logger.debug("Data channel is open, flushing queued messages")
            while self._message_queue:
                message = self._message_queue.pop(0)
                self._data_channel.send(message)

        @self._data_channel.on("message")
        async def on_message(message: str):
            logger.debug(f"Data channel message received: {message}")
            try:
                if message.startswith("ping"):
                    self._last_received_time = time.time()
                    return

                json_message = json.loads(message)
                # Handle app messages directly.
                if json_message.get("type") != SIGNALLING_TYPE:
                    await self._call_event_handler("app-message", json_message)
                    return

                # We have a signalling message. Handle it carefully here to avoid
                # calling the parent's brittle handler with unexpected types.
                signal = json_message.get("message", {})
                signal_type = signal.get("type")

                logger.debug(f"Handling signalling message of type: {signal_type}")

                if signal_type == "peerLeft":
                    logger.info("Peer has left the connection, closing.")
                    asyncio.create_task(self.disconnect())
                elif signal_type:
                    # For all other known signal types, we can safely pass them
                    # to the parent's handler.
                    super(LocalCovalDailyConnection, self)._handle_signalling_message(signal)
                else:
                    logger.warning(f"Received unknown signalling message: {signal}")

            except Exception as e:
                logger.exception(f"Error processing data channel message {message}: {e}")

        logger.debug("Adding audio and video transceivers")
        self._pc.addTransceiver("audio", direction="sendrecv")
        self._pc.addTransceiver("video", direction="sendrecv")

        logger.debug("Creating offer as initiator")
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        self._offer = self._pc.localDescription

        # Now, wait for ICE gathering to complete.
        logger.info("Waiting for ICE gathering to complete...")
        try:
            await asyncio.wait_for(self._ice_gathering_complete.wait(), timeout=10.0)
            offer_ice_count = self._pc.localDescription.sdp.count("a=candidate")
            logger.info(f"ICE gathering complete. Offer has {offer_ice_count} candidates.")
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out. Proceeding with partial offer.")

        return self.get_offer()

    def get_offer(self):
        """Return the generated offer."""
        if not self._offer:
            return None

        return {
            "sdp": self._offer.sdp,
            "type": self._offer.type,
            "pc_id": self._pc_id,
        }

    async def process_answer(self, sdp: str, type: str):
        """Process an SDP answer received in response to our offer."""
        if not self._initiator:
            logger.warning("Attempted to process an answer while not in initiator mode")
            return

        if not self._offer:
            logger.warning("No offer has been created yet")
            return

        logger.debug("Processing answer as initiator")
        self._answer = RTCSessionDescription(sdp=sdp, type=type)
        await self._pc.setRemoteDescription(self._answer)

        # Force transceivers to sendrecv mode
        self.force_transceivers_to_send_recv()

    async def connect(self):
        """Connect to the Coval cloud agent, ensuring the logic only runs once."""
        async with self._connect_lock:
            # Check the real connection state from the base class. If another task
            # has already connected, there's nothing more to do.
            if self.is_connected():
                logger.debug("Connection is already established, skipping redundant connect call.")
                return

            try:
                logger.info("Starting Local Coval Daily connection...")

                # 1. Create the complete offer (with all ICE candidates).
                offer = await self.create_offer()

                # 2. Create a signaling session with the complete offer.
                logger.info("Creating signaling session...")
                self._session_id = await self._create_signaling_session(offer["sdp"])
                logger.info(f"Signaling session created: {self._session_id}")

                # 3. Launch the evaluation to start the cloud agent.
                logger.info("Launching evaluation...")
                self._evaluation_run_id = await self._launch_evaluation()
                logger.info(f"Evaluation launched: {self._evaluation_run_id}")

                # 4. Poll for the cloud agent's SDP answer.
                logger.info("Polling for cloud agent's SDP answer...")
                answer_sdp = await self._poll_for_cloud_answer()
                logger.info("Received SDP answer from cloud agent.")

                # 5. Process the answer.
                await self.process_answer(sdp=answer_sdp, type="answer")

                # 6. Finalize the connection using the base class method.
                await super().connect()

                logger.info("Local Coval Daily connection logic finished.")

            except Exception as e:
                logger.error(f"Failed to establish Coval connection: {e}")
                raise

    def _setup_listeners(self):
        """Set up WebRTC event listeners."""
        super()._setup_listeners()

        @self._pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            logger.debug(f"ICE gathering state is {self._pc.iceGatheringState}")
            if self._pc.iceGatheringState == "complete":
                self._ice_gathering_complete.set()

    async def _create_signaling_session(self, sdp_offer: str) -> str:
        """Create a signaling session with the SDP offer."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "action": "CREATE_SESSION",
            "organization_id": self._organization_id,
            "agent_id": self._agent_id,
            "sdp_offer": sdp_offer,
            "client_metadata": self._client_metadata,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._signaling_base_url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["session_id"]

    async def _launch_evaluation(self) -> str:
        """Launch an evaluation to start the cloud agent."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        # Use a deep copy to avoid modifying the original user-provided config.
        payload = copy.deepcopy(self._evaluation_config)

        # Ensure the model object and its type are set correctly for this transport.
        model = payload.setdefault("model", {})
        model["type"] = "MODEL_TYPE_DAILY_LOCAL"

        # Ensure the model config object exists and inject the session ID.
        config = model.setdefault("config", {})
        config["signaling_session_id"] = self._session_id
        config["agent_id"] = self._agent_id

        # If in dev mode, ensure dev_id is present, as required by local servers.
        if self._dev_mode:
            payload["dev_id"] = self._evaluation_config.get("dev_id", "default_dev_id")

        logger.debug(f"Launching evaluation with payload: {json.dumps(payload, indent=2)}")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._launch_evaluation_url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["run_id"]

    async def _poll_for_cloud_answer(self) -> str:
        """Poll for the SDP answer from the cloud agent."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "session_id": self._session_id,
        }

        logger.debug(f"Polling session status with payload: {json.dumps(payload)}")

        max_attempts = 180  # 3 minutes
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_attempts):
                try:
                    async with session.post(
                        self._session_status_base_url, headers=headers, json=payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()

                            if "sdp_answer" in data and data["sdp_answer"]:
                                logger.info(f"Received cloud answer after {attempt}s")
                                return data["sdp_answer"]
                            elif "error" in data:
                                raise Exception(f"Session error: {data['error']}")
                            else:
                                logger.debug(f"Waiting for cloud answer... ({attempt}s)")
                        else:
                            logger.warning(f"Session status check failed: {response.status}")

                except Exception as e:
                    logger.warning(f"Session status poll failed: {e}")

                await asyncio.sleep(1.0)

        raise Exception(f"Timeout waiting for cloud SDP answer after {max_attempts}s.")


class LocalCovalDailyTransport(SmallWebRTCTransport):
    """WebRTC transport for Local Coval Daily connections."""

    def __init__(
        self,
        *,
        params: TransportParams,
        api_key: str,
        evaluation_config: Dict[str, Any],
        client_metadata: Optional[dict] = None,
        ice_servers: Optional[list] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        # Determine which URLs to use based on the deployment_type
        deployment_type = evaluation_config.get("deployment_type", "prod")
        dev_mode = deployment_type == "dev"

        if dev_mode:
            logger.info("Coval transport running in DEV mode. Using localhost URLs.")
            signaling_base_url = "http://localhost:8080/covaldailysignaling"
            session_status_base_url = "http://localhost:8080/covalsessionstatus"
            launch_evaluation_url = "http://localhost:8080/launchevaluation"
        else:
            logger.info("Coval transport running in PROD mode.")
            signaling_base_url = "https://api.coval.dev/covaldailysignaling"
            session_status_base_url = "https://api.coval.dev/covalsessionstatus"
            launch_evaluation_url = "https://api.coval.dev/launchEvaluation"

        # Create the Coval WebRTC connection
        connection = LocalCovalDailyConnection(
            api_key=api_key,
            signaling_base_url=signaling_base_url,
            session_status_base_url=session_status_base_url,
            launch_evaluation_url=launch_evaluation_url,
            evaluation_config=evaluation_config,
            dev_mode=dev_mode,
            client_metadata=client_metadata,
            ice_servers=ice_servers,
        )

        super().__init__(
            webrtc_connection=connection,
            params=params,
            input_name=input_name,
            output_name=output_name,
        )

        # Override the default output transport with our custom one.
        self._output = LocalCovalDailyOutputTransport(self._client, params, name=self._output_name)

    @property
    def session_id(self) -> Optional[str]:
        # We need to access the connection from the client.
        return self._client._webrtc_connection._session_id

    @property
    def evaluation_run_id(self) -> Optional[str]:
        # We need to access the connection from the client.
        return self._client._webrtc_connection._evaluation_run_id
