# Set default goal
.DEFAULT_GOAL := run

# Run command to start the Streamlit app
run:
	@echo "Running the Streamlit app..."
	streamlit run app.py
