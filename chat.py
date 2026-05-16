import asyncio
import gradium

client = gradium.client.GradiumClient(api_key="gsk_98643412cd7a5c8b3270e5acb2d3f348c628833bbf38f46ce9434075082c23aa")

# 1. A mock async generator to simulate streaming real PCM audio chunks
async def mock_audio_stream():
    # In a real app, this would read from a microphone stream or a file
    # yielding raw binary bytes like: yield chunk_bytes
    mock_pcm_chunks = [b"\x00\x00" * 500, b"\x11\x11" * 500, b"\x22\x22" * 500]
    
    for chunk in mock_pcm_chunks:
        yield chunk
        await asyncio.sleep(0.1)  # Simulating a slight delay between speech frames

async def transcribe(audio_source):
    async with client.stt_realtime(
        model_name="default",
        input_format="pcm",
    ) as stt:
        async def producer():
            # audio_source will be an async generator of PCM chunks
            async for chunk in audio_source:
                await stt.send_audio(chunk)
            await stt.send_eos()

        async def consumer():
            transcript = []
            async for msg in stt:
                if msg["type"] == "text":
                    transcript.append(msg["text"])
                elif msg["type"] == "end_of_stream":
                    break
            return " ".join(transcript)

        _, transcribed_text = await asyncio.gather(producer(), consumer())
        return transcribed_text

if __name__ == "__main__":
    audio_stream = mock_audio_stream()
    try:
        transcribed_text = asyncio.run(transcribe(audio_stream))
        print("Transcribed:", transcribed_text)
    except Exception as e:
        print(f"Pipeline Error: {e}")
