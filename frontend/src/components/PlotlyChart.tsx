import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { Chart } from "../types";

const dark = () => window.matchMedia?.("(prefers-color-scheme: dark)").matches;

export function PlotlyChart({ chart, height = 340 }: { chart: Chart; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const isDark = dark();
    const layout = {
      ...(chart.figure.layout as object),
      height,
      autosize: true,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, system-ui, sans-serif", color: isDark ? "#e5e7eb" : "#0f172a", size: 12 },
      xaxis: { ...((chart.figure.layout as any)?.xaxis ?? {}), gridcolor: isDark ? "#1f2a3d" : "#e2e8f0" },
      yaxis: { ...((chart.figure.layout as any)?.yaxis ?? {}), gridcolor: isDark ? "#1f2a3d" : "#e2e8f0" },
      template: undefined,
    };
    Plotly.react(ref.current, chart.figure.data as Plotly.Data[], layout as Partial<Plotly.Layout>, {
      responsive: true,
      displaylogo: false,
    });
    const el = ref.current;
    return () => {
      Plotly.purge(el);
    };
  }, [chart, height]);
  return <div ref={ref} className="chart-box" />;
}
