"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  Shield,
  Cpu,
  Radio,
  Wifi,
  Phone,
  Upload,
  AlertTriangle,
  Activity,
  Bell,
  Clock,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { PageContainer } from "@/components/layout/PageContainer";

const statusItems = [
  {
    label: "AI Engine",
    value: "V2",
    icon: Cpu,
    status: "ready" as const,
  },
  {
    label: "Detection Engine",
    value: "Ready",
    icon: Radio,
    status: "ready" as const,
  },
  {
    label: "Connection",
    value: "Offline",
    icon: Wifi,
    status: "offline" as const,
  },
];

const overviewCards = [
  {
    label: "Calls Analyzed",
    value: "—",
    icon: Phone,
    hint: "Appears after backend integration",
  },
  {
    label: "Potential Threats",
    value: "—",
    icon: AlertTriangle,
    hint: "Appears after backend integration",
  },
  {
    label: "High Risk Events",
    value: "—",
    icon: Bell,
    hint: "Appears after backend integration",
  },
];

const statusColors = {
  ready: "bg-primary",
  offline: "bg-muted-foreground/50",
};

export default function DashboardPage() {
  return (
    <>
      <Navbar />
      <PageContainer>
        <div className="space-y-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <h1 className="text-2xl font-bold tracking-tight">
              Security Dashboard
            </h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Overview of the Morph voice security system. Real-time data will
              appear once the backend is connected.
            </p>
          </motion.div>

          {/* System Status */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.05 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h2 className="text-sm font-semibold mb-4">System Status</h2>
            <div className="grid gap-4 sm:grid-cols-3">
              {statusItems.map((item) => (
                <div
                  key={item.label}
                  className="flex items-center gap-3 rounded-lg bg-background/50 px-4 py-3"
                >
                  <item.icon className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-muted-foreground">{item.label}</p>
                    <p className="text-sm font-medium">{item.value}</p>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div
                      className={`h-2 w-2 rounded-full ${statusColors[item.status]}`}
                    />
                    <span className="text-[10px] text-muted-foreground capitalize">
                      {item.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Detection Overview */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="grid gap-4 sm:grid-cols-3"
          >
            {overviewCards.map((card) => (
              <div
                key={card.label}
                className="rounded-xl border border-border bg-card p-5"
              >
                <div className="flex items-center justify-between mb-3">
                  <card.icon className="h-4 w-4 text-muted-foreground" />
                </div>
                <p className="text-2xl font-bold text-muted-foreground/50">
                  {card.value}
                </p>
                <p className="text-sm font-medium mt-1">{card.label}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {card.hint}
                </p>
              </div>
            ))}
          </motion.div>

          {/* Risk Overview — Empty State */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.15 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold">Risk Overview</h2>
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
                No data
              </span>
            </div>
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-16">
              <Activity className="h-8 w-8 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">
                Risk distribution will appear here once detections are
                processed.
              </p>
              <p className="text-xs text-muted-foreground/60 mt-1">
                Connected to backend API
              </p>
            </div>
          </motion.div>

          {/* Recent Detection Activity — Empty State */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <div className="flex items-center gap-2 mb-4">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold">
                Recent Detection Activity
              </h2>
            </div>
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-12">
              <Shield className="h-8 w-8 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">
                No detection events yet.
              </p>
              <p className="text-xs text-muted-foreground/60 mt-1 max-w-xs text-center">
                Events will appear here after live detection is connected and
                calls are analyzed.
              </p>
            </div>
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.25 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h2 className="text-sm font-semibold mb-4">Quick Actions</h2>
            <div className="grid gap-3 sm:grid-cols-2">
              <Link
                href="/call"
                className="flex items-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-3 transition-colors hover:border-primary/20 hover:bg-secondary"
              >
                <Phone className="h-4 w-4 text-primary" />
                <div>
                  <p className="text-sm font-medium">Start Live Call</p>
                  <p className="text-xs text-muted-foreground">
                    Begin real-time voice analysis
                  </p>
                </div>
              </Link>
              <Link
                href="/detection"
                className="flex items-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-3 transition-colors hover:border-primary/20 hover:bg-secondary"
              >
                <Upload className="h-4 w-4 text-primary" />
                <div>
                  <p className="text-sm font-medium">Analyze Voice</p>
                  <p className="text-xs text-muted-foreground">
                    Upload a recording for detection
                  </p>
                </div>
              </Link>
            </div>
          </motion.div>
        </div>
      </PageContainer>
      <Footer />
    </>
  );
}
