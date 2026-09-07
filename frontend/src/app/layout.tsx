import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";
import "./globals.css";

export const metadata: Metadata = {
  title: "Morph — Synthetic Voice Detection",
  description:
    "Real-time AI-powered detection of synthetic and deepfake audio communications.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen bg-grid antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem={false}
        >
          <div className="min-h-screen bg-grid">{children}</div>
          <Toaster
            position="bottom-right"
            toastOptions={{
              style: {
                background: "#111827",
                border: "1px solid #1e293b",
                color: "#e2e8f0",
              },
            }}
          />
        </ThemeProvider>
      </body>
    </html>
  );
}
