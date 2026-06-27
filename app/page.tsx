"use client";

import React, { useEffect, useRef, useState } from "react";
import { Camera, Shield, Cpu, Activity, Info } from "lucide-react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

interface WebcamStreamState {
  status: "inactive" | "requesting" | "active" | "error";
  permission: "not_requested" | "granted" | "denied";
  error: string | null;
}

interface HandDetectionState {
  hand_label: "Left" | "Right" | "Unknown";
  confidence: number;
  landmark_count: number;
  motion: string | null;
  detected_at: Date;
}

const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],

  // Index finger
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],

  // Middle finger
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],

  // Ring finger
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],

  // Pinky
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],

  // Palm connections
  [5, 9],
  [9, 13],
  [13, 17],
];

function drawHandSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number }[],
  width: number,
  height: number
) {
  // Draw connecting lines first
  ctx.strokeStyle = "#A78BFA";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const start = landmarks[startIdx];
    const end = landmarks[endIdx];

    if (!start || !end) continue;

    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  }

  // Draw landmark dots on top
  ctx.fillStyle = "#7C3AED";

  for (const point of landmarks) {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Draw smaller white center dots
  ctx.fillStyle = "#FFFFFF";

  for (const point of landmarks) {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, 2, 0, 2 * Math.PI);
    ctx.fill();
  }
}

export default function HandDetectorPage() {
  const [streamState, setStreamState] = useState<WebcamStreamState>({
    status: "inactive",
    permission: "not_requested",
    error: null,
  });

  const [detections, setDetections] = useState<HandDetectionState[]>([]);
  const [isModelReady, setIsModelReady] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    async function initTracking() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
        );

        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });

        setIsModelReady(true);
      } catch (err) {
        console.error("Failed to initialize MediaPipe Hand Landmarker:", err);

        setStreamState({
          status: "error",
          permission: "not_requested",
          error: "Failed to load MediaPipe hand tracking model.",
        });
      }
    }

    initTracking();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      const video = videoRef.current;

      if (video?.srcObject) {
        const stream = video.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startWebcam = async () => {
    if (!isModelReady) {
      setStreamState({
        status: "error",
        permission: "not_requested",
        error: "MediaPipe model is still loading. Wait a few seconds, then try again.",
      });
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setStreamState({
        status: "error",
        permission: "denied",
        error: "Camera API is not available. Use localhost or HTTPS.",
      });
      return;
    }

    setStreamState({
      status: "requesting",
      permission: "not_requested",
      error: null,
    });

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: "user",
        },
        audio: false,
      });

      const video = videoRef.current;

      if (!video) return;

      video.srcObject = stream;

      video.onloadedmetadata = async () => {
        await video.play();

        setStreamState({
          status: "active",
          permission: "granted",
          error: null,
        });

        startDetectionLoop();
      };
    } catch (err: any) {
      setStreamState({
        status: "error",
        permission: "denied",
        error: err.message || "Failed to start webcam.",
      });
    }
  };

  const startDetectionLoop = () => {
    const processFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const landmarker = landmarkerRef.current;

      if (!video || !canvas || !landmarker || video.readyState < 2) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
        return;
      }

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
        return;
      }

      const ctx = canvas.getContext("2d");

      if (!ctx) {
        animationFrameRef.current = requestAnimationFrame(processFrame);
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const timestamp = performance.now();
      const result = landmarker.detectForVideo(video, timestamp);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const currentDetections: HandDetectionState[] = [];

      if (result.landmarks && result.landmarks.length > 0) {
        result.landmarks.forEach((landmarks, index) => {
          let handedness: "Left" | "Right" | "Unknown" = "Unknown";
          let confidence = 0;

          if (result.handednesses && result.handednesses[index]) {
            const info = result.handednesses[index][0];

            if (info.categoryName === "Left") {
              handedness = "Left";
            } else if (info.categoryName === "Right") {
              handedness = "Right";
            }

            confidence = info.score;
          }

          currentDetections.push({
            hand_label: handedness,
            confidence,
            landmark_count: landmarks.length,
            motion: "Stable Static View",
            detected_at: new Date(),
          });

          drawHandSkeleton(ctx, landmarks, canvas.width, canvas.height);
        });
      }

      setDetections(currentDetections);

      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    animationFrameRef.current = requestAnimationFrame(processFrame);
  };

  const hand_count = streamState.status === "active" ? detections.length : 0;
  const left_hand_count = detections.filter((d) => d.hand_label === "Left").length;
  const right_hand_count = detections.filter((d) => d.hand_label === "Right").length;

  const getDetectedHandsText = () => {
    if (hand_count === 0) return "No hand detected";
    if (left_hand_count === 1 && right_hand_count === 0) return "Detected: Left hand";
    if (left_hand_count === 0 && right_hand_count === 1) return "Detected: Right hand";
    if (left_hand_count === 1 && right_hand_count === 1) return "Detected: Two hands";
    return "Dynamic State Tracking Active";
  };

  return (
    <main className="min-h-screen bg-background text-gray-100 p-6 flex flex-col items-center">
      <header className="w-full max-w-6xl mb-8 flex flex-col md:flex-row items-start md:items-center justify-between border-b border-purple-900/40 pb-4 gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-primary to-accent flex items-center gap-2">
            <Cpu className="w-6 h-6 text-primary animate-pulse" />
            Hand Motion Webcam v1
          </h1>
          <p className="text-xs text-gray-400 mt-1 uppercase">
            jorj project #3
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs text-gray-400">
          <div className="flex items-center gap-2 bg-surface px-3 py-1.5 rounded-large border border-gray-800">
            <Shield className="w-4 h-4 text-accent" />
            
          </div>
        </div>
      </header>

      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 flex flex-col bg-surface rounded-large border border-gray-800 shadow-soft overflow-hidden">
          <div className="p-4 bg-gray-900/60 border-b border-gray-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Camera className="w-4 h-4 text-primary" />
              <span className="text-sm font-semibold tracking-wide uppercase">
                Stream Feed
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  streamState.status === "active"
                    ? "bg-green-500 animate-ping"
                    : streamState.status === "error"
                    ? "bg-red-500"
                    : "bg-yellow-500"
                }`}
              />
              <span className="text-xs font-mono uppercase text-gray-300">
                {streamState.status}
              </span>
            </div>
          </div>

          <div className="relative bg-black flex-1 min-h-[400px] flex items-center justify-center p-2">
            {streamState.status !== "active" ? (
              <div className="text-center p-6 max-w-md z-10">
                <p className="text-sm text-gray-400 mb-4">
                  Stream request access is required to parse core local hardware image
                  descriptors via browser context.
                </p>

                <button
                  onClick={startWebcam}
                  disabled={streamState.status === "requesting" || !isModelReady}
                  className="px-5 py-2.5 bg-primary hover:bg-secondary text-white text-xs font-bold rounded-large transition-all tracking-widest disabled:opacity-50 uppercase border border-accent/20"
                >
                  {!isModelReady
                    ? "Loading AI Model..."
                    : streamState.status === "requesting"
                    ? "Requesting Hardware Context..."
                    : "Initialize Webcam"}
                </button>

                {streamState.error && (
                  <p className="mt-4 text-xs bg-red-950/40 text-red-400 p-3 rounded-large border border-red-900/50">
                    ERR: {streamState.error}
                  </p>
                )}
              </div>
            ) : null}

            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover opacity-40 pointer-events-none"
              style={{ transform: "scaleX(-1)" }}
            />

            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
              style={{ transform: "scaleX(-1)" }}
            />
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="p-5 bg-surface rounded-large border border-gray-800 shadow-soft">
            <h2 className="text-xs uppercase text-accent font-bold tracking-widest mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4 text-primary" />
              Core Describe Engine
            </h2>

            <div className="bg-background/90 p-4 rounded-large border border-purple-900/20 text-center">
              <span className="text-sm text-gray-400 block uppercase tracking-wider mb-1 font-sans">
                Current Logic Rule Match
              </span>

              <div className="text-lg font-bold text-white tracking-wide font-mono">
                &gt; {getDetectedHandsText()}
              </div>
            </div>
          </div>

          <div className="p-5 bg-surface rounded-large border border-gray-800 shadow-soft flex-1">
            <h2 className="text-xs uppercase text-gray-400 font-bold tracking-widest mb-4 flex items-center gap-2">
              <Info className="w-4 h-4 text-primary" />
              Compute Telemetry Metrics
            </h2>

            <div className="space-y-4 font-mono text-xs">
              <div className="p-3 bg-background rounded-md border border-gray-800/80 flex justify-between items-center">
                <span className="text-gray-400">Hand Count:</span>
                <span className="text-base font-bold text-accent">{hand_count}</span>
              </div>

              <div className="p-3 bg-background rounded-md border border-gray-800/80 flex justify-between items-center">
                <span className="text-gray-400">Left-hand Count:</span>
                <span className="text-base font-bold text-primary">
                  {left_hand_count}
                </span>
              </div>

              <div className="p-3 bg-background rounded-md border border-gray-800/80 flex justify-between items-center">
                <span className="text-gray-400">Right-hand Count:</span>
                <span className="text-base font-bold text-primary">
                  {right_hand_count}
                </span>
              </div>
            </div>

            <div className="mt-6">
              <h3 className="text-[10px] uppercase text-gray-500 tracking-wider mb-2 font-bold">
                Volatile Frame Detections Struct
              </h3>

              <div className="bg-background rounded-large p-3 border border-gray-800 max-h-[160px] overflow-y-auto space-y-2">
                {detections.length === 0 ? (
                  <p className="text-[11px] text-gray-600 italic">
                    No historical frames currently evaluated in tracking matrix context.
                  </p>
                ) : (
                  detections.map((det, idx) => (
                    <div
                      key={idx}
                      className="text-[11px] border-b border-gray-900 pb-2 last:border-none last:pb-0"
                    >
                      <div className="flex justify-between text-gray-300">
                        <span>
                          Label:{" "}
                          <strong className="text-accent">{det.hand_label}</strong>
                        </span>
                        <span>Conf: {(det.confidence * 100).toFixed(1)}%</span>
                      </div>

                      <div className="text-gray-500 text-[10px] mt-0.5 flex justify-between">
                        <span>Points: {det.landmark_count}/21</span>
                        <span>{det.motion}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}