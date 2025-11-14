def format_output(predictions):
    formatted_output = [{"prediction": pred} for pred in predictions]
    return formatted_output

def format_analysis(analysis):
    formatted_analysis = {"analysis": analysis}
    return formatted_analysis
