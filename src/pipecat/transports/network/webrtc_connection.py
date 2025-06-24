#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
from typing import Any, Literal, Optional, Union, List

from loguru import logger
from pydantic import BaseModel, TypeAdapter

from pipecat.utils.base_object import BaseObject

try:
    from aiortc import (
        MediaStreamTrack,
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.rtcrtpreceiver import RemoteStreamTrack
    from av.frame import Frame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the SmallWebRTC, you need to `pip install pipecat-ai[webrtc]`.")
    raise Exception(f"Missing module: {e}")

SIGNALLING_TYPE = "signalling"
AUDIO_TRANSCEIVER_INDEX = 0
VIDEO_TRANSCEIVER_INDEX = 1


class TrackStatusMessage(BaseModel):
    type: Literal["trackStatus"]
    receiver_index: int
    enabled: bool


class RenegotiateMessage(BaseModel):
    type: Literal["renegotiate"] = "renegotiate"


class PeerLeftMessage(BaseModel):
    type: Literal["peerLeft"] = "peerLeft"


class SignallingMessage:
    Inbound = Union[
        TrackStatusMessage, RenegotiateMessage, PeerLeftMessage
    ]  # in case we need to add new messages in the future
    outbound = Union[RenegotiateMessage, PeerLeftMessage]


class SmallWebRTCTrack:
    def __init__(self, track: MediaStreamTrack):
        self._track = track
        self._enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    async def discard_old_frames(self):
        remote_track = self._track
        if isinstance(remote_track, RemoteStreamTrack):
            if not hasattr(remote_track, "_queue") or not isinstance(
                remote_track._queue, asyncio.Queue
            ):
                print("Warning: _queue does not exist or has changed in aiortc.")
                return
            logger.debug("Discarding old frames")
            while not remote_track._queue.empty():
                remote_track._queue.get_nowait()  # Remove the oldest frame
                remote_track._queue.task_done()

    async def recv(self) -> Optional[Frame]:
        if not self._enabled:
            return None
        return await self._track.recv()

    def __getattr__(self, name):
        # Forward other attribute/method calls to the underlying track
        return getattr(self._track, name)


# Alias so we don't need to expose RTCIceServer
IceServer = RTCIceServer


class SmallWebRTCConnection(BaseObject):
    def __init__(
        self, ice_servers: Optional[List[Union[str, RTCIceServer]]] = None, initiator: bool = False
    ):
        super().__init__()
        if not ice_servers:
            self.ice_servers: List[IceServer] = []
        elif all(isinstance(s, IceServer) for s in ice_servers):
            self.ice_servers = ice_servers
        elif all(isinstance(s, str) for s in ice_servers):
            self.ice_servers = [IceServer(urls=s) for s in ice_servers]
        elif all(isinstance(s, dict) for s in ice_servers):
            self.ice_servers = [IceServer(**s) for s in ice_servers]
        else:
            raise TypeError("ice_servers must be either List[str] or List[RTCIceServer]")
        self._connect_invoked = False
        self._track_map = {}
        self._track_getters = {
            AUDIO_TRANSCEIVER_INDEX: self.audio_input_track,
            VIDEO_TRANSCEIVER_INDEX: self.video_input_track,
        }
        self._initiator = initiator
        self._offer = None
        self._data_channel_name = "data"

        self._initialize()

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("app-message")
        self._register_event_handler("track-started")
        self._register_event_handler("track-ended")
        # connection states
        self._register_event_handler("connecting")
        self._register_event_handler("connected")
        self._register_event_handler("disconnected")
        self._register_event_handler("closed")
        self._register_event_handler("failed")
        self._register_event_handler("new")
        self._register_event_handler("offer-created")

    @property
    def pc(self) -> RTCPeerConnection:
        return self._pc

    @property
    def pc_id(self) -> str:
        return self._pc_id

    def _initialize(self):
        logger.debug("Initializing new peer connection")
        rtc_config = RTCConfiguration(iceServers=self.ice_servers)

        self._answer: Optional[RTCSessionDescription] = None
        self._offer = None
        self._pc = RTCPeerConnection(rtc_config)
        self._pc_id = self.name
        self._setup_listeners()
        self._data_channel = None
        self._renegotiation_in_progress = False
        self._last_received_time = None
        self._message_queue = []
        self._pending_app_messages = []

    def _setup_listeners(self):
        # Only set up datachannel listener when we're not the initiator
        # When we're initiator, we create the data channel explicitly
        if not self._initiator:

            @self._pc.on("datachannel")
            def on_datachannel(channel):
                logger.debug(f"Data channel created: {channel}")
                self._data_channel = channel

                # Flush queued messages once the data channel is open
                @channel.on("open")
                async def on_open():
                    logger.debug("Data channel is open, flushing queued messages")
                    while self._message_queue:
                        message = self._message_queue.pop(0)
                        logger.debug(f"Sending message to data channel: {message}")
                        self._data_channel.send(message)

                @channel.on("message")
                async def on_message(message):
                    try:
                        # aiortc does not provide any way so we can be aware when we are disconnected,
                        # so we are using this keep alive message as a way to implement that
                        if isinstance(message, str) and message.startswith("ping"):
                            self._last_received_time = time.time()
                        else:
                            if self.is_connected():
                                await self._call_event_handler("app-message", json_message)
                            else:
                                logger.debug("Client not connected. Queuing app-message.")
                                self._pending_app_messages.append(json_message)
                    except Exception as e:
                        logger.exception(f"Error parsing JSON message {message}, {e}")

        # Despite the fact that aiortc provides this listener, they don't have a status for "disconnected"
        # So, in case we loose connection, this event will not be triggered
        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange():
            await self._handle_new_connection_state()

        # Despite the fact that aiortc provides this listener, they don't have a status for "disconnected"
        # So, in case we loose connection, this event will not be triggered
        @self._pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.debug(
                f"ICE connection state is {self._pc.iceConnectionState}, connection is {self._pc.connectionState}"
            )

        @self._pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            logger.debug(f"ICE gathering state is {self._pc.iceGatheringState}")

        @self._pc.on("track")
        async def on_track(track):
            logger.debug(f"Track {track.kind} received")
            await self._call_event_handler("track-started", track)

            @track.on("ended")
            async def on_ended():
                logger.debug(f"Track {track.kind} ended")
                await self._call_event_handler("track-ended", track)

    async def _create_answer(self, sdp: str, type: str):
        offer = RTCSessionDescription(sdp=sdp, type=type)
        await self._pc.setRemoteDescription(offer)

        # For some reason, aiortc is not respecting the SDP for the transceivers to be sendrcv
        # so we are basically forcing it to act this way
        self.force_transceivers_to_send_recv()

        # this answer does not contain the ice candidates, which will be gathered later, after the setLocalDescription
        logger.debug(f"Creating answer")
        local_answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(local_answer)
        logger.debug(f"Setting the answer after the local description is created")
        self._answer = self._pc.localDescription

    def get_offer(self):
        """Return the generated offer when in initiator mode."""
        if not self._offer:
            return None

        return {
            "sdp": self._offer.sdp,
            "type": self._offer.type,
            "pc_id": self._pc_id,
        }

    async def initialize(self, sdp: str, type: str):
        """Initialize the connection with remote SDP.

        In responder mode: set remote offer and create answer
        In initiator mode: raises an exception (use create_offer instead)

        Raises:
            ValueError: If called in initiator mode
        """
        if self._initiator:
            raise RuntimeError("In initiator mode, use create_offer() instead of initialize()")
        await self._create_answer(sdp, type)

    async def connect(self):
        """Establish the connection.

        In responder mode: connects using the previously initialized SDP
        In initiator mode: requires an offer to have been created and an answer processed

        Raises:
            RuntimeError: If called in initiator mode without a processed answer
        """
        self._connect_invoked = True

        # If initiator mode, we must have both offer and answer before connecting
        if self._initiator:
            if not self._offer:
                raise RuntimeError(
                    "Cannot connect in initiator mode without creating an offer first"
                )
            if not self._answer:
                raise RuntimeError(
                    "Cannot connect in initiator mode without processing an answer first"
                )

        # If we already connected, trigger again the connected event
        if self.is_connected():
            await self._call_event_handler("connected")
            logger.debug("Flushing pending app-messages")
            for message in self._pending_app_messages:
                await self._call_event_handler("app-message", message)
            # We are renegotiating here, because likely we have loose the first video frames
            # and aiortc does not handle that pretty well.
            video_input_track = self.video_input_track()
            if video_input_track:
                await self.video_input_track().discard_old_frames()
            self.ask_to_renegotiate()

    async def renegotiate(self, sdp: str, type: str, restart_pc: bool = False):
        logger.debug(f"Renegotiating {self._pc_id}")

        if restart_pc:
            await self._call_event_handler("disconnected")
            logger.debug("Closing old peer connection")
            # removing the listeners to prevent the bot from closing
            self._pc.remove_all_listeners()
            await self._close()
            # we are initializing a new peer connection in this case.
            self._initialize()

        await self._create_answer(sdp, type)

        # Maybe we should refactor to receive a message from the client side when the renegotiation is completed.
        # or look at the peer connection listeners
        # but this is good enough for now for testing.
        async def delayed_task():
            await asyncio.sleep(2)
            self._renegotiation_in_progress = False

        asyncio.create_task(delayed_task())

    def force_transceivers_to_send_recv(self):
        for transceiver in self._pc.getTransceivers():
            transceiver.direction = "sendrecv"
            # logger.debug(
            #    f"Transceiver: {transceiver}, Mid: {transceiver.mid}, Direction: {transceiver.direction}"
            # )
            # logger.debug(f"Sender track: {transceiver.sender.track}")

    def replace_audio_track(self, track):
        logger.debug(f"Replacing audio track {track.kind}")
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        transceivers = self._pc.getTransceivers()
        if len(transceivers) > 0 and transceivers[0].sender:
            transceivers[0].sender.replaceTrack(track)
        else:
            logger.warning("Audio transceiver not found. Cannot replace audio track.")

    def replace_video_track(self, track):
        logger.debug(f"Replacing video track {track.kind}")
        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        transceivers = self._pc.getTransceivers()
        if len(transceivers) > 1 and transceivers[1].sender:
            transceivers[1].sender.replaceTrack(track)
        else:
            logger.warning("Video transceiver not found. Cannot replace video track.")

    async def disconnect(self):
        self.send_app_message({"type": SIGNALLING_TYPE, "message": PeerLeftMessage().model_dump()})
        await self._close()

    async def _close(self):
        if self._pc:
            await self._pc.close()
        self._message_queue.clear()
        self._pending_app_messages.clear()
        self._track_map = {}

    def get_answer(self):
        if not self._answer:
            return None

        return {
            "sdp": self._answer.sdp,
            "type": self._answer.type,
            "pc_id": self._pc_id,
        }

    async def _handle_new_connection_state(self):
        state = self._pc.connectionState
        if state == "connected" and not self._connect_invoked:
            # We are going to wait until the pipeline is ready before triggering the event
            return
        logger.debug(f"Connection state changed to: {state}")
        await self._call_event_handler(state)
        if state == "failed":
            logger.warning("Connection failed, closing peer connection.")
            await self._close()

    # Despite the fact that aiortc provides this listener, they don't have a status for "disconnected"
    # So, there is no advantage in looking at self._pc.connectionState
    # That is why we are trying to keep our own state
    def is_connected(self):
        # If the small webrtc transport has never invoked to connect
        # we are acting like if we are not connected
        if not self._connect_invoked:
            return False

        if self._last_received_time is None:
            # if we have never received a message, it is probably because the client has not created a data channel
            # so we are going to trust aiortc in this case
            return self._pc.connectionState == "connected"
        # Checks if the last received ping was within the last 3 seconds.
        return (time.time() - self._last_received_time) < 3

    def audio_input_track(self):
        if self._track_map.get(AUDIO_TRANSCEIVER_INDEX):
            return self._track_map[AUDIO_TRANSCEIVER_INDEX]

        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        transceivers = self._pc.getTransceivers()
        if len(transceivers) == 0 or not transceivers[AUDIO_TRANSCEIVER_INDEX].receiver:
            logger.warning("No audio transceiver is available")
            return None

        track = transceivers[AUDIO_TRANSCEIVER_INDEX].receiver.track
        audio_track = SmallWebRTCTrack(track) if track else None
        self._track_map[AUDIO_TRANSCEIVER_INDEX] = audio_track
        return audio_track

    def video_input_track(self):
        if self._track_map.get(VIDEO_TRANSCEIVER_INDEX):
            return self._track_map[VIDEO_TRANSCEIVER_INDEX]

        # Transceivers always appear in creation-order for both peers
        # For now we are only considering that we are going to have 02 transceivers,
        # one for audio and one for video
        transceivers = self._pc.getTransceivers()
        if len(transceivers) <= 1 or not transceivers[VIDEO_TRANSCEIVER_INDEX].receiver:
            logger.warning("No video transceiver is available")
            return None

        track = transceivers[VIDEO_TRANSCEIVER_INDEX].receiver.track
        video_track = SmallWebRTCTrack(track) if track else None
        self._track_map[VIDEO_TRANSCEIVER_INDEX] = video_track
        return video_track

    def send_app_message(self, message: Any):
        json_message = json.dumps(message)
        if self._data_channel and self._data_channel.readyState == "open":
            self._data_channel.send(json_message)
        else:
            logger.debug("Data channel not ready, queuing message")
            self._message_queue.append(json_message)

    def ask_to_renegotiate(self):
        if self._renegotiation_in_progress:
            return

        self._renegotiation_in_progress = True
        self.send_app_message(
            {"type": SIGNALLING_TYPE, "message": RenegotiateMessage().model_dump()}
        )

    def _handle_signalling_message(self, message):
        logger.debug(f"Signalling message received: {message}")
        inbound_adapter = TypeAdapter(SignallingMessage.Inbound)
        signalling_message = inbound_adapter.validate_python(message)
        match signalling_message:
            case TrackStatusMessage():
                track = (
                    self._track_getters.get(signalling_message.receiver_index) or (lambda: None)
                )()
                if track:
                    track.set_enabled(signalling_message.enabled)
            case PeerLeftMessage():
                logger.debug("Received peer left message")
            case RenegotiateMessage():
                logger.debug("Received renegotiation message")

    async def create_offer(self):
        """Creates an SDP offer when in initiator mode.

        Includes both data channel and audio/video transceivers.

        Returns:
            dict: The SDP offer containing sdp, type and pc_id if successful
            None: If not in initiator mode

        Raises:
            RuntimeError: If offer creation fails
        """
        if not self._initiator:
            raise RuntimeError("Attempted to create an offer while not in initiator mode")

        # Create a data channel before generating the offer
        # This ensures the SDP will include data channel information
        self._data_channel = self._pc.createDataChannel(self._data_channel_name)

        # Set up the data channel event handlers
        @self._data_channel.on("open")
        async def on_open():
            logger.debug("Data channel is open, flushing queued messages")
            while self._message_queue:
                message = self._message_queue.pop(0)
                logger.debug(f"Flushing message to data channel: {message}")
                self._data_channel.send(message)

        @self._data_channel.on("message")
        async def on_message(message):
            logger.debug(f"Data channel message received: {message}")
            try:
                if isinstance(message, str) and message.startswith("ping"):
                    self._last_received_time = time.time()
                else:
                    json_message = json.loads(message)
                    if json_message["type"] == SIGNALLING_TYPE and json_message.get("message"):
                        self._handle_signalling_message(json_message["message"])
                    else:
                        await self._call_event_handler("app-message", json_message)
            except Exception as e:
                logger.exception(f"Error parsing JSON message {message}, {e}")

        # Add transceivers for audio and video
        logger.debug("Adding audio transceivers")
        self._pc.addTransceiver("audio", direction="sendrecv")
        self._pc.addTransceiver("video", direction="sendrecv")

        # Create offer
        logger.debug("Creating offer as initiator")
        try:
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)
            self._offer = self._pc.localDescription

            # Notify that an offer has been created
            await self._call_event_handler("offer-created", self.get_offer())

            return self.get_offer()
        except Exception as e:
            logger.exception("Failed to create offer")
            raise RuntimeError(f"Failed to create offer: {str(e)}")

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
