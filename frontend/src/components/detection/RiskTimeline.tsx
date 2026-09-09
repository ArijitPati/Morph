"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  DotProps,
} from "recharts";
import type { WindowResult } from "@/types/detection";

interface RiskTimelineProps {
  windows: WindowResult[];
  threshold?: number;
}

export function RiskTimeline({ windows, threshold = 0.5 }: RiskTimelineProps) {
  if (!windows.length) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-8">
        <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider">
          Window-by-window analysis
        </p>
        <p className="text-[10px] text-muted-foreground/60 mt-0.5">
          Appears after backend analysis
        </p>
      </div>
    );
  }

  const data = windows.map((w) => ({
    window: `W${w.window}`,
    idx: w.window,
    window_index: w.window_index,
    fake_prob: Math.round(w.fake_probability * 1000) / 1000,
    fake_pct: Math.round(w.fake_probability * 1000) / 10,
    label: w.label_str,
    start: w.start_sec,
    end: w.end_sec,
    is_partial: w.is_partial,
  }));

  // Custom dot — stable key per window (window_index is backend-unique, never re-used within a result)
  const Dot = (props: DotProps & { payload?: { label: string; idx: number; window_index?: number }; index?: number }) => {
    const { payload, index } = props as unknown as {
      payload: { label: string; idx: number; window_index?: number };
      index: number;
    };
    const color = payload?.label === "FAKE" ? "#ef4444" : "#06d6a0";
    // window_index is the canonical unique id; fallback to idx/index for legacy data
    const stableIdx = payload?.window_index ?? payload?.idx ?? index ?? 0;
    return (
      <circle
        key={`dot-${stableIdx}-${props.cx}-${props.cy}`}
        cx={props.cx}
        cy={props.cy}
        r={4}
        fill={color}
        stroke="white"
        strokeWidth={1}
      />
    );
  };

  return (
    <div className="space-y-3">
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 12, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
            <XAxis
              dataKey="window"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              axisLine={false}
              tickLine={false}
              width={36}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "P(FAKE)"]}
              labelFormatter={(label: string, payload: unknown) => {
                const p = (payload as Array<{ payload: { start: number; end: number; is_partial: boolean } }>)?.[0]?.payload;
                if (!p) return label;
                return `${label} [${p.start.toFixed(1)}s–${p.end.toFixed(1)}s]${p.is_partial ? " partial" : ""}`;
              }}
            />
            <ReferenceLine
              y={threshold}
              stroke="rgba(234,179,8,0.5)"
              strokeDasharray="6 3"
              label={{
                value: "threshold",
                position: "insideTopRight",
                fontSize: 9,
                fill: "rgba(234,179,8,0.8)",
              }}
            />
            <Line
              type="monotone"
              dataKey="fake_prob"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={Dot as unknown as boolean}
              activeDot={{ r: 5, strokeWidth: 1.5 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Window chips — window_index is stable unique id */}
      <div className="flex flex-wrap gap-1.5">
        {windows.map((w) => (
          <div
            key={`chip-${w.window_index}-${w.start_sec}`}
            className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] font-medium ${
              w.label_str === "FAKE"
                ? "border-danger/30 bg-danger/10 text-danger"
                : "border-primary/20 bg-primary/10 text-primary"
            }`}
            title={`${w.start_sec}s – ${w.end_sec}s · P(FAKE)=${(w.fake_probability * 100).toFixed(1)}%${w.is_partial ? " · partial" : ""}`}
          >
            <span>W{w.window}</span>
            <span className="opacity-60">{(w.fake_probability * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
