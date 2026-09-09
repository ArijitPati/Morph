export interface SignalingJoinMessage {
  type: "join";
  payload: { room: string };
}

export interface SignalingOfferMessage {
  type: "offer";
  payload: { room: string; sdp: RTCSessionDescriptionInit };
}

export interface SignalingAnswerMessage {
  type: "answer";
  payload: { room: string; sdp: RTCSessionDescriptionInit };
}

export interface SignalingIceMessage {
  type: "ice-candidate";
  payload: { room: string; candidate: RTCIceCandidateInit };
}

export interface SignalingLeaveMessage {
  type: "leave";
  payload: { room: string };
}

export type SignalingOutgoing =
  | SignalingJoinMessage
  | SignalingOfferMessage
  | SignalingAnswerMessage
  | SignalingIceMessage
  | SignalingLeaveMessage;

export interface SignalingJoinedMessage {
  type: "joined";
  payload: { room: string; peers: number };
}

export interface SignalingPeerJoinedMessage {
  type: "peer-joined";
  payload: { room: string; peers: number };
}

export interface SignalingPeerLeftMessage {
  type: "peer-left";
  payload: { room: string; peers: number };
}

export interface SignalingIncomingOffer {
  type: "offer";
  payload: { room: string; sdp: RTCSessionDescriptionInit };
}

export interface SignalingIncomingAnswer {
  type: "answer";
  payload: { room: string; sdp: RTCSessionDescriptionInit };
}

export interface SignalingIncomingIce {
  type: "ice-candidate";
  payload: { room: string; candidate: RTCIceCandidateInit };
}

export interface SignalingError {
  type: "error";
  payload: { code: string; message: string };
}

export type SignalingIncoming =
  | SignalingJoinedMessage
  | SignalingPeerJoinedMessage
  | SignalingPeerLeftMessage
  | SignalingIncomingOffer
  | SignalingIncomingAnswer
  | SignalingIncomingIce
  | SignalingError;
