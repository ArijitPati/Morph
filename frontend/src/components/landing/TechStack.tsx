"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const categories = [
  {
    label: "AI Engine",
    items: ["XGBoost", "132 Acoustic Features", "Python", "scikit-learn"],
  },
  {
    label: "Audio Processing",
    items: ["Librosa", "NumPy", "SciPy", "SoundFile"],
  },
  {
    label: "Communication",
    items: ["WebRTC", "WebSocket"],
  },
  {
    label: "Backend",
    items: ["FastAPI", "Uvicorn"],
  },
  {
    label: "Frontend",
    items: ["Next.js", "TypeScript", "Tailwind CSS", "Framer Motion"],
  },
];

export function TechStack() {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section ref={ref} className="py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-12"
        >
          <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">
            Built for Real-Time Voice Security
          </h2>
          <p className="mt-2 max-w-xl text-muted-foreground">
            The technologies powering the Morph detection platform.
          </p>
        </motion.div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {categories.map((cat, i) => (
            <motion.div
              key={cat.label}
              initial={{ opacity: 0, y: 16 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.4, delay: 0.08 * i }}
              className="rounded-xl border border-border bg-card p-5"
            >
              <h3 className="text-xs font-semibold uppercase tracking-wider text-primary">
                {cat.label}
              </h3>
              <ul className="mt-3 space-y-1.5">
                {cat.items.map((item) => (
                  <li
                    key={item}
                    className="text-sm text-muted-foreground"
                  >
                    {item}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
