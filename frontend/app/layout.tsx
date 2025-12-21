import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Multimodal Document Chat",
  description: "Chat with your documents using AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-gray-50">
          {/* Navigation */}
          <nav className="bg-white shadow-sm border-b">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex space-x-8">
                  <Link href="/" className="flex items-center text-gray-900 hover:text-blue-600">
                    <span className="font-bold text-xl">📄 DocChat</span>
                  </Link>
                  <Link href="/" className="flex items-center text-gray-700 hover:text-blue-600">
                    Documents
                  </Link>
                  <Link href="/upload" className="flex items-center text-gray-700 hover:text-blue-600">
                    Upload
                  </Link>
                  <Link href="/chat" className="flex items-center text-gray-700 hover:text-blue-600">
                    Chat
                  </Link>
                </div>
              </div>
            </div>
          </nav>

          {/* Main Content */}
          <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
