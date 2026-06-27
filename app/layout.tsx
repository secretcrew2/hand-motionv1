import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "HandMotionWebcam Dashboard",
  description: "Real-time AI Hand-Tracking System",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark bg-background text-gray-100 antialiased">
      <body className="font-mono">{children}</body>
    </html>
  );
}