from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from fastapi import FastAPI
from pydantic import BaseModel

# Define the model from the Hugging Face library
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Create an instance of the FastAPI class
app = FastAPI()

# Define the Pydantic model for input validation
class Story(BaseModel):
    story: str

# Output the sentiment scores
@app.get("/story-analysis", response_model=dict)
def polarity_scores_roberta(story: Story):   
    # Encode the given story
    encoded_story = tokenizer(story.story, return_tensors='pt') 
    
    # Get the model output
    output = model(**encoded_story)
    
    # Compute scores
    scores = output.logits[0].detach().numpy()  # Use .logits for Hugging Face models
    scores = softmax(scores)
    
    # Turn scores into a dictionary
    scores_dict = {
        'negative': float(scores[0]),
        'neutral': float(scores[1]),
        'positive': float(scores[2])
    }
    
    return scores_dict
