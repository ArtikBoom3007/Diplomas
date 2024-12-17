from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyedflib import highlevel
from loguru import logger
import json
import os
import random
from pathlib import Path

app = FastAPI()

class ProcessRequest(BaseModel):
    input_file: str
    output_dir: str = "output"

# Function to process input and generate output
def process_files(input_data, output_dir):
    output_results = []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for entry in input_data:
        input_filename = entry["input_filename"]
        signals, signal_headers, _ = highlevel.read_edf(input_filename)
        logger.info(f"Signal from {input_filename} readed. Signal shape: {signals.shape}")

        # Generate random values for result and predict
        result = random.randint(0, 1)
        predict = random.uniform(0, 1)

        # Define output filename
        output_filename = Path(output_dir) / f"out_{Path(input_filename).stem}.json"

        # Create the output JSON structure
        output_data = {
            "result": result,
            "predict": predict
        }

        # Write the output file
        with open(output_filename, "w") as f:
            json.dump(output_data, f, indent=4)

        # Append the result to the overall output list
        output_results.append({
            "input_filename": input_filename,
            "output_filename": str(output_filename)
        })

    return output_results

@app.post("/process")
def process_request(request: ProcessRequest):
    try:
        # Read input JSON
        with open(request.input_file, "r") as f:
            input_data = json.load(f)

        # Process files and get results
        results = process_files(input_data, request.output_dir)

        # Return the results as JSON
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
