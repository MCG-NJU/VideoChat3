import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export const metadata: Metadata = {
  title: "VideoChat3 — Generalist Video MLLM",
  description: "A token-efficient, fully open video MLLM for fine-grained motion, long-form reasoning, temporal grounding, and live streaming.",
  icons: { icon: `${basePath}/parrot.png`, shortcut: `${basePath}/parrot.png` },
  openGraph: {
    title: "VideoChat3 — Generalist Video MLLM",
    description: "Token-efficient and fully open across fine-grained motion, long-form reasoning, temporal grounding, and live streaming.",
    type: "website",
  },
  twitter: { card: "summary_large_image", title: "VideoChat3 — Generalist Video MLLM", description: "Token-efficient and fully open video understanding across four temporal regimes." },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
