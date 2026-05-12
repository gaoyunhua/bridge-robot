# Bridge Game Web UI

Play bridge with AI opponents or watch AI compete!

## Features

- 🤖 **4 AI Players**: Watch AI play against itself
- 🧑‍🤝‍🤖 **3 AI + 1 Human**: Play against AI
- 🎴 **Full Bridge Game**: Bidding and play phases
- 📊 **Real-time Game State**: See the game unfold
- 📈 **AI with DDS Support**: Uses trained policy model

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
cd src/ui
./start.sh
# OR
python app.py
```

3. Open your browser: http://localhost:5000

## Game Modes

### 4 AI Players (Watch)
- Click the button and watch AI compete
- Each AI move plays automatically
- Great for observing AI behavior

### 3 AI + 1 Human
- Play as one player against 3 AI
- When it's your turn, you'll see bid/card options
- AI will play automatically on their turns

## Project Structure

```
src/ui/
├── app.py              # Flask server & game logic
├── templates/
│   └── index.html     # Web UI
├── requirements.txt    # Python dependencies
└── start.sh           # Startup script
```

## Model

The UI uses the trained policy model from `checkpoints/policy_model_v2.pt`

If you don't have a model, the AI will play randomly.

## Development

To modify the UI:
- HTML/CSS: Edit `templates/index.html`
- Backend: Edit `app.py`
- Styling: Modify the `<style>` section in the HTML
