# Sample payloads

The JSON files here show the exact response **format** of `/ask`. They were produced by running
the reasoning engine on hand-made detections, so the numbers are illustrative.

Before submitting, replace them with real responses from your trained model:

    curl -s -X POST http://localhost:8000/detect -F "file=@samples/notes_example.jpg" > samples/detect_response.json
    curl -s -X POST http://localhost:8000/ask -F "question=How much money is in this photo?" -F "file=@samples/notes_example.jpg" > samples/ask_total.json
