"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";

interface DocumentDetail {
  id: number;
  filename: string;
  upload_date: string;
  status: string;
  error_message?: string;
  total_pages: number;
  text_chunks: number;
  images: Array<{
    id: number;
    url: string;
    page: number;
    caption?: string;
    width: number;
    height: number;
  }>;
  tables: Array<{
    id: number;
    url: string;
    page: number;
    caption?: string;
    rows: number;
    columns: number;
    data?: any;
  }>;
}

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDocument();
  }, [params.id]);

  const fetchDocument = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/documents/${params.id}`);
      const data = await response.json();
      setDocument(data);
    } catch (error) {
      console.error('Error fetching document:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
        <p className="mt-2 text-gray-600">Loading document...</p>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-600">Document not found</p>
        <Link href="/" className="text-blue-600 hover:text-blue-700 mt-4 inline-block">
          Back to documents
        </Link>
      </div>
    );
  }

  return (
    <div className="px-4 sm:px-0">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{document.filename}</h1>
            <p className="text-sm text-gray-500 mt-1">
              Uploaded: {new Date(document.upload_date).toLocaleDateString()}
            </p>
          </div>
          <div className="flex space-x-3">
            <Link
              href={`/chat?document=${document.id}`}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
            >
              Chat with Document
            </Link>
            <button
              onClick={() => router.push('/')}
              className="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300"
            >
              Back
            </button>
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="bg-white shadow rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Processing Status</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500">Status</p>
            <p className={`text-lg font-semibold ${
              document.status === 'completed' ? 'text-green-600' :
              document.status === 'processing' ? 'text-yellow-600' :
              document.status === 'error' ? 'text-red-600' :
              'text-gray-600'
            }`}>
              {document.status}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Pages</p>
            <p className="text-lg font-semibold text-gray-900">{document.total_pages}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Text Chunks</p>
            <p className="text-lg font-semibold text-gray-900">{document.text_chunks}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Media</p>
            <p className="text-lg font-semibold text-gray-900">
              {document.images.length} images, {document.tables.length} tables
            </p>
          </div>
        </div>
        
        {document.error_message && (
          <div className="mt-4 p-4 bg-red-50 rounded-lg">
            <p className="text-sm text-red-800">{document.error_message}</p>
          </div>
        )}
      </div>

      {/* Images */}
      {document.images.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">
            Extracted Images ({document.images.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {document.images.map((image) => (
              <div key={image.id} className="border rounded-lg p-4">
                <img
                  src={`http://localhost:8000${image.url}`}
                  alt={image.caption || 'Document image'}
                  className="w-full rounded mb-2"
                />
                <p className="text-sm text-gray-600">
                  {image.caption || `Image from page ${image.page}`}
                </p>
                <p className="text-xs text-gray-500">
                  Page {image.page} • {image.width}x{image.height}px
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tables */}
      {document.tables.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">
            Extracted Tables ({document.tables.length})
          </h2>
          <div className="space-y-4">
            {document.tables.map((table) => (
              <div key={table.id} className="border rounded-lg p-4">
                <img
                  src={`http://localhost:8000${table.url}`}
                  alt={table.caption || 'Document table'}
                  className="w-full rounded mb-2"
                />
                <p className="text-sm text-gray-600">
                  {table.caption || `Table from page ${table.page}`}
                </p>
                <p className="text-xs text-gray-500">
                  Page {table.page} • {table.rows} rows × {table.columns} columns
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
