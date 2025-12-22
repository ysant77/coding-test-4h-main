"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { useSearchParams } from "next/navigation";

interface SourceText {
  type: "text";
  page?: number | null;
  score?: number | null;
  content: string;
}

interface SourceMedia {
  type: "image" | "table";
  url: string;                 // may be relative (/uploads/...) or absolute (http...)
  caption?: string | null;
  page?: number | null;
}

interface SourceTableStructured {
  type: "table_structured";
  caption?: string | null;
  page?: number | null;
  rows: string[][];
}

type Source = SourceText | SourceMedia | SourceTableStructured | any;

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  created_at: string;
}

function joinUrl(base: string, maybeRelative: string): string {
  if (!maybeRelative) return "";
  // already absolute
  if (/^https?:\/\//i.test(maybeRelative)) return maybeRelative;
  // ensure exactly one slash between
  const b = base.replace(/\/+$/, "");
  const p = maybeRelative.startsWith("/") ? maybeRelative : `/${maybeRelative}`;
  return `${b}${p}`;
}

function safeNum(n: any): number | null {
  const x = Number(n);
  return Number.isFinite(x) ? x : null;
}

export default function ChatPage() {
  const searchParams = useSearchParams();
  const documentIdStr = searchParams.get("document");
  const documentId = documentIdStr ? parseInt(documentIdStr) : null;

  const API_BASE = useMemo(() => {
    // Prefer env var. Fallback to localhost.
    // In Next, you must define NEXT_PUBLIC_API_BASE in .env.local to customize.
    return process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  }, []);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const sendMessageRest = async (userMessage: string) => {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        conversation_id: conversationId,
        document_id: documentId,
      }),
    });

    // handle non-2xx
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`HTTP ${response.status}: ${text || response.statusText}`);
    }

    const data = await response.json();

    if (!conversationId) setConversationId(data.conversation_id);

    const assistantMessage: Message = {
      id: data.message_id ?? Date.now(),
      role: "assistant",
      content: data.answer ?? "",
      sources: data.sources ?? [],
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, assistantMessage]);
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setLoading(true);

    // Add user message immediately
    const tempUserMessage: Message = {
      id: Date.now(),
      role: "user",
      content: userMessage,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMessage]);

    try {
      await sendMessageRest(userMessage);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          content: "Sorry, I encountered an error processing your message.",
          created_at: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const renderStructuredTable = (rows: string[][]) => {
    if (!rows || rows.length === 0) return <p className="text-xs text-gray-500">(empty table)</p>;
    const head = rows[0] || [];
    const body = rows.slice(1);

    return (
      <div className="overflow-x-auto border rounded bg-white">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="bg-gray-50">
              {head.map((h, i) => (
                <th key={i} className="text-left p-2 border-b font-semibold text-gray-700">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((r, ri) => (
              <tr key={ri} className="odd:bg-white even:bg-gray-50">
                {r.map((c, ci) => (
                  <td key={ci} className="p-2 border-b text-gray-700 align-top">
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-0 h-[calc(100vh-12rem)]">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">
        Chat with Document
        {documentId && (
          <span className="text-sm text-gray-500 ml-2">(Document #{documentId})</span>
        )}
      </h1>

      <div className="bg-white shadow rounded-lg flex flex-col h-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <p>Start a conversation by asking a question about the document.</p>
              <p className="text-sm mt-2">Try asking about images, tables, or specific content.</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-900"
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 space-y-3">
                      {msg.sources.map((source: any, sidx: number) => {
                        const type = source?.type;

                        // Media (image/table) as PNG/JPG
                        if (type === "image" || type === "table") {
                          const mediaUrl = joinUrl(API_BASE, source.url);
                          const caption = source.caption || (type === "image" ? "Image" : "Table");
                          const page = safeNum(source.page);

                          return (
                            <div key={sidx} className="border-t border-gray-300 pt-3">
                              <p className="text-xs text-gray-600 mb-2">
                                {caption}
                                {page !== null ? ` (Page ${page})` : ""}
                              </p>
                              <img
                                src={mediaUrl}
                                alt={caption}
                                className="max-w-full rounded border"
                                loading="lazy"
                              />
                            </div>
                          );
                        }

                        // Structured table (rows)
                        if (type === "table_structured" && Array.isArray(source.rows)) {
                          const caption = source.caption || "Table";
                          const page = safeNum(source.page);
                          return (
                            <div key={sidx} className="border-t border-gray-300 pt-3">
                              <p className="text-xs text-gray-600 mb-2">
                                {caption}
                                {page !== null ? ` (Page ${page})` : ""}
                              </p>
                              {renderStructuredTable(source.rows)}
                            </div>
                          );
                        }

                        // Text chunks
                        if (type === "text") {
                          const page = safeNum(source.page);
                          const score = safeNum(source.score);
                          return (
                            <div
                              key={sidx}
                              className="border-t border-gray-300 pt-3 text-xs text-gray-600 bg-white p-2 rounded"
                            >
                              <p className="font-semibold mb-1">
                                Source{page !== null ? ` (Page ${page})` : ""}{score !== null ? `, Score: ${score.toFixed(2)}` : ""}
                              </p>
                              <p className="whitespace-pre-wrap">{source.content}</p>
                            </div>
                          );
                        }

                        // Fallback (unknown)
                        return (
                          <div key={sidx} className="border-t border-gray-300 pt-3 text-xs text-gray-600">
                            <p className="font-semibold mb-1">Source</p>
                            <pre className="bg-white p-2 rounded overflow-x-auto">
                              {JSON.stringify(source, null, 2)}
                            </pre>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg p-4">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t p-4">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Ask a question about the document..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || loading}
              className={`px-6 py-2 rounded-lg font-medium ${
                input.trim() && !loading
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`}
            >
              Send
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            API: {API_BASE}
          </p>
        </div>
      </div>
    </div>
  );
}
