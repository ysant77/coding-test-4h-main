"use client";

import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";
import Image from "next/image";

interface Message {
  id: number;
  role: string;
  content: string;
  sources?: any[];
  created_at: string;
}

export default function ChatPage() {
  const searchParams = useSearchParams();
  const documentId = searchParams.get('document');
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input;
    setInput('');
    setLoading(true);

    // Add user message to UI immediately
    const tempUserMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: userMessage,
      created_at: new Date().toISOString()
    };
    setMessages(prev => [...prev, tempUserMessage]);

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_id: conversationId,
          document_id: documentId ? parseInt(documentId) : null
        }),
      });

      const data = await response.json();
      
      if (!conversationId) {
        setConversationId(data.conversation_id);
      }

      // Add assistant message
      const assistantMessage: Message = {
        id: data.message_id,
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        created_at: new Date().toISOString()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your message.',
        created_at: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-0 h-[calc(100vh-12rem)]">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">
        Chat with Document
        {documentId && <span className="text-sm text-gray-500 ml-2">(Document #{documentId})</span>}
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
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                  
                  {/* Display sources (images, tables, text) */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 space-y-3">
                      {msg.sources.map((source, sidx) => (
                        <div key={sidx} className="border-t border-gray-300 pt-3">
                          {source.type === 'image' && (
                            <div>
                              <p className="text-xs text-gray-600 mb-2">
                                {source.caption || 'Image'}
                              </p>
                              <img
                                src={`http://localhost:8000${source.url}`}
                                alt={source.caption || 'Document image'}
                                className="max-w-full rounded"
                              />
                            </div>
                          )}
                          
                          {source.type === 'table' && (
                            <div>
                              <p className="text-xs text-gray-600 mb-2">
                                {source.caption || 'Table'}
                              </p>
                              <img
                                src={`http://localhost:8000${source.url}`}
                                alt={source.caption || 'Document table'}
                                className="max-w-full rounded"
                              />
                            </div>
                          )}
                          
                          {source.type === 'text' && (
                            <div className="text-xs text-gray-600 bg-white p-2 rounded">
                              <p className="font-semibold mb-1">
                                Source (Page {source.page}, Score: {source.score?.toFixed(2)})
                              </p>
                              <p className="line-clamp-3">{source.content}</p>
                            </div>
                          )}
                        </div>
                      ))}
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
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
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
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about the document..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || loading}
              className={`px-6 py-2 rounded-lg font-medium ${
                input.trim() && !loading
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
