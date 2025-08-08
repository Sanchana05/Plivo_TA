from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transcribe import transcribe_file
from diarize import diarize_file
from image_caption import caption_image
from docs_parser import parse_and_summarize
from typing import Optional
import tempfile, shutil, os

router = APIRouter()

@router.post("/conversation")
async def conversation_skill(file: UploadFile = File(...)):
    # save file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    # transcription
    transcript, segments = transcribe_file(tmp.name)
    # diarization using segments + audio
    diarized = diarize_file(tmp.name, segments)
    # optional summary (backend can call OpenAI if API key set)
    from docs_parser import summarize_text
    summary = summarize_text(transcript)
    os.unlink(tmp.name)
    return JSONResponse({"transcript": transcript, "diarization": diarized, "summary": summary})


@router.post("/image")
async def image_skill(file: UploadFile = File(...)):
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as out:
        out.write(await file.read())
    caption = caption_image(tmp_path)
    os.unlink(tmp_path)
    return {"caption": caption}

@router.post("/summarize")
async def summarize_skill(file: UploadFile = File(None), url: Optional[str] = Form(None)):
    # if file given -> parse; else fetch url
    if file:
        tmp_path = f"/tmp/{file.filename}"
        with open(tmp_path, "wb") as out:
            out.write(await file.read())
        summary = parse_and_summarize(tmp_path)
        os.unlink(tmp_path)
    elif url:
        summary = parse_and_summarize(url=url)
    else:
        return {"error": "Provide file or url"}
    return {"summary": summary}
