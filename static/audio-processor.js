class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.position = 0;
    this.pending = [];
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch) {
      const ratio = sampleRate / 16000;

      while (this.position < ch.length) {
        this.pending.push(ch[Math.floor(this.position)]);
        this.position += ratio;
      }
      this.position -= ch.length;

      while (this.pending.length >= 800) {
        const chunk = this.pending.splice(0, 800);
        const out = new Int16Array(chunk.length);
        for (let i = 0; i < chunk.length; i++) {
          const sample = Math.max(-1, Math.min(1, chunk[i]));
          out[i] = sample < 0 ? sample * 32768 : sample * 32767;
        }
        this.port.postMessage(out.buffer, [out.buffer]);
      }
    }
    return true;
  }
}
registerProcessor("pcm-processor", PCMProcessor);
