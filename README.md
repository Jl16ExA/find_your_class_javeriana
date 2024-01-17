# "Encuentra tu clase Javeriana" Streamlit Application

## Overview
"Encuentra tu clase Javeriana" is an innovative Streamlit application designed to help users find classes related to their interests at Javeriana University. Utilizing state-of-the-art NLP techniques, this app processes user queries to match them with relevant class offerings.

## Key Features
- **Class Search**: Enter the name or topic of interest to find related classes.
- **Customizable Results**: Adjust the number of classes displayed using an intuitive slider.
- **Cosine Similarity**: Employs cosine similarity to determine the relevance of classes to your query.
- **Data Visualization**: Results are presented in an easy-to-read format.

## Free and Open Source
This application is completely free to use. We employ a small open-source model (`intfloat/multilingual-e5-small`) for natural language processing tasks. Moreover, we use numpy arrays for vector embeddings, avoiding the need for a more complex vector database. This choice underscores our commitment to efficiency and accessibility.

## Inspiration and Community
We encourage users to draw inspiration from this project. It's a testament to the power of open-source tools in creating impactful applications. We're excited to see how others can adapt and extend our work.

## Support and Collaboration
If you have any questions or wish to collaborate, feel free to reach out via my social media platforms. I am always eager to discuss and help.

- **Developer**: Juan David López
- **GitHub**: [Jl16ExA](https://github.com/Jl16ExA)
- **LinkedIn**: [Juan David López Becerra](https://www.linkedin.com/in/juan-david-lopez-becerra-5048271bb/)
- **Twitter**: [@JLopez_160](https://twitter.com/JLopez_160)

## Installation and Usage
1. **Clone the Repository**: `git clone https://github.com/Jl16ExA/encuentra-tu-clase-javeriana.git`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run the Application**: `streamlit run app.py`

## Contributing
We welcome contributions! If you have suggestions or want to improve the app, please submit a pull request or open an issue in the GitHub repository.
