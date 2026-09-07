import { cn } from "@/lib/utils";

type StatusType = "operational" | "degraded" | "down" | "analyzing";

const statusConfig: Record<
  StatusType,
  { label: string; dotClass: string; textClass: string }
> = {
  operational: {
    label: "Operational",
    dotClass: "bg-safe",
    textClass: "text-safe",
  },
  degraded: {
    label: "Degraded",
    dotClass: "bg-warning",
    textClass: "text-warning",
  },
  down: {
    label: "Down",
    dotClass: "bg-danger",
    textClass: "text-danger",
  },
  analyzing: {
    label: "Analyzing",
    dotClass: "bg-accent",
    textClass: "text-accent",
  },
};

interface StatusBadgeProps {
  status: StatusType;
  className?: string;
  showLabel?: boolean;
}

export function StatusBadge({
  status,
  className,
  showLabel = true,
}: StatusBadgeProps) {
  const config = statusConfig[status];

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="relative">
        <div
          className={cn("h-2 w-2 rounded-full", config.dotClass)}
        />
        <div
          className={cn(
            "absolute inset-0 h-2 w-2 rounded-full animate-ping opacity-75",
            config.dotClass
          )}
        />
      </div>
      {showLabel && (
        <span className={cn("text-xs font-medium", config.textClass)}>
          {config.label}
        </span>
      )}
    </div>
  );
}
