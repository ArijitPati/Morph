"use client";

import { useEffect, useRef } from "react";

const BAR_COUNT = 48;
const CENTER = BAR_COUNT / 2;

export function VoiceVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animFrame: number;
    let time = 0;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Capture ctx for the draw closure
    const c = ctx;

    function draw() {
      time += 0.02;
      c.clearRect(0, 0, w, h);

      const barWidth = w / BAR_COUNT;

      for (let i = 0; i < BAR_COUNT; i++) {
        const distFromCenter = Math.abs(i - CENTER) / CENTER;
        const falloff = 1 - distFromCenter * 0.6;

        const amp1 = Math.sin(time * 1.2 + i * 0.3) * 0.3;
        const amp2 = Math.sin(time * 0.8 + i * 0.5) * 0.2;
        const amp3 = Math.sin(time * 1.6 + i * 0.15) * 0.15;
        const amplitude = (amp1 + amp2 + amp3) * falloff;

        const baseHeight = 4;
        const barH = baseHeight + Math.abs(amplitude) * (h * 0.6);
        const x = i * barWidth + barWidth * 0.2;
        const barW = barWidth * 0.6;
        const y = (h - barH) / 2;

        // Color: green near center, fading outward
        const alpha = 0.15 + (1 - distFromCenter) * 0.55;
        const green = Math.round(180 + (1 - distFromCenter) * 34);
        c.fillStyle = `rgba(6, ${green}, 160, ${alpha})`;

        c.beginPath();
        c.roundRect(x, y, barW, barH, 1.5);
        c.fill();
      }

      // Center indicator line
      const cx = w / 2;
      c.strokeStyle = "rgba(6, 214, 160, 0.25)";
      c.lineWidth = 1;
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(cx, h * 0.15);
      c.lineTo(cx, h * 0.85);
      c.stroke();
      c.setLineDash([]);

      // Top label
      c.fillStyle = "rgba(148, 163, 184, 0.6)";
      c.font = "10px system-ui, sans-serif";
      c.textAlign = "center";
      c.fillText("VOICE STREAM", cx, h * 0.1);

      // Bottom label
      c.fillText("MORPH ANALYSIS", cx, h * 0.95);

      animFrame = requestAnimationFrame(draw);
    }

    draw();

    return () => cancelAnimationFrame(animFrame);
  }, []);

  return (
    <div className="relative w-full max-w-md">
      <div className="rounded-xl border border-border bg-card p-1">
        <canvas
          ref={canvasRef}
          className="h-64 w-full sm:h-72"
          style={{ imageRendering: "auto" }}
        />
      </div>
      <div className="mt-2 flex items-center justify-center gap-6">
        <div className="flex items-center gap-1.5">
          <div className="h-1 w-4 rounded-full bg-primary/60" />
          <span className="text-[10px] text-muted-foreground">
            Acoustic Features
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="h-1 w-4 rounded-full bg-primary/30" />
          <span className="text-[10px] text-muted-foreground">
            Spectral Profile
          </span>
        </div>
      </div>
    </div>
  );
}
