import React, { useState, useRef } from 'react';
import { Mic, Square, Upload } from 'lucide-react';

const encodeWAV = (samples, sampleRate) => {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  
  const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // 1 channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // blockAlign
  view.setUint16(34, 16, true); // bits/sample
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
};

const convertToWav = async (blob) => {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  const channelData = audioBuffer.getChannelData(0);
  return encodeWAV(channelData, audioBuffer.sampleRate);
};

export default function AudioControl({ onSubmit, disabled }) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const rawBlob = new Blob(audioChunksRef.current);
        audioChunksRef.current = [];
        
        try {
          const wavBlob = await convertToWav(rawBlob);
          onSubmit(wavBlob);
        } catch (e) {
          console.error("Failed to convert audio to WAV:", e);
          onSubmit(rawBlob);
        }
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      audioChunksRef.current = [];
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      alert("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      onSubmit(file);
    }
  };

  return (
    <div className="controls-container">
      {!isRecording ? (
        <button 
          className="btn btn-primary" 
          onClick={startRecording}
          disabled={disabled}
        >
          <Mic size={20} /> Record Audio
        </button>
      ) : (
        <button 
          className="btn btn-danger recording" 
          onClick={stopRecording}
          disabled={disabled}
        >
          <Square size={20} /> Stop Recording
        </button>
      )}

      <label className={`btn btn-secondary ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
        <Upload size={20} /> Upload File
        <input 
          type="file" 
          accept="audio/*" 
          className="hidden-input" 
          onChange={handleFileUpload}
          disabled={disabled}
        />
      </label>
    </div>
  );
}
