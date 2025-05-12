"use client";

import Link from 'next/link';
import { ArrowDownTrayIcon, CameraIcon } from '@heroicons/react/24/outline';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center">
      {/* Hero section */}
      <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48 bg-gradient-to-br from-blue-900 to-purple-900">
        <div className="container px-4 md:px-6 mx-auto">
          <div className="flex flex-col items-center space-y-4 text-center">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none text-white">
                Car Model Classification
              </h1>
              <p className="mx-auto max-w-[700px] text-gray-200 md:text-xl">
                Upload an image of a car and our AI will identify the model using advanced ResNet technology.
              </p>
            </div>
            <div className="space-x-4">
              <Link
                href="/classify"
                className="inline-flex h-12 items-center justify-center rounded-md bg-white px-8 text-lg font-medium text-blue-900 shadow transition-colors hover:bg-blue-50"
              >
                <CameraIcon className="mr-2 h-5 w-5" />
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-gray-100">
        <div className="container px-4 md:px-6 mx-auto">
          <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-3">
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="bg-blue-100 p-4 rounded-full">
                <CameraIcon className="h-8 w-8 text-blue-700" />
              </div>
              <h3 className="text-xl font-bold">Upload Your Image</h3>
              <p className="text-gray-500">
                Simply upload an image of any car and our system will preprocess it.
              </p>
            </div>
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="bg-purple-100 p-4 rounded-full">
                <ArrowDownTrayIcon className="h-8 w-8 text-purple-700" />
              </div>
              <h3 className="text-xl font-bold">Choose Your Model</h3>
              <p className="text-gray-500">
                Select from multiple ResNet architectures with different capabilities.
              </p>
            </div>
            <div className="flex flex-col items-center text-center space-y-4">
              <div className="bg-green-100 p-4 rounded-full">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-8 w-8 text-green-700">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
                </svg>
              </div>
              <h3 className="text-xl font-bold">Get Accurate Results</h3>
              <p className="text-gray-500">
                Receive precise classification with confidence scores for the detected car model.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-blue-900 text-white">
        <div className="container px-4 md:px-6 mx-auto">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter md:text-4xl/tight">
                Ready to Identify Your Car?
              </h2>
              <p className="mx-auto max-w-[600px] text-gray-200 md:text-xl/relaxed">
                Our advanced AI can recognize hundreds of car models with impressive accuracy.
              </p>
            </div>
            <div className="space-x-4">
              <Link
                href="/classify"
                className="inline-flex h-12 items-center justify-center rounded-md bg-white px-8 text-lg font-medium text-blue-900 shadow transition-colors hover:bg-blue-50"
              >
                Try It Now
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
} 