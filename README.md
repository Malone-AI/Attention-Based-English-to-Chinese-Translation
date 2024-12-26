# Attention-Based English to Chinese Translation Application

[English](README.md) | [中文](README.zh-CN.md)

## Project Introduction
This is an English-to-Chinese translation application built with Flask, aimed at exploring the application of `Attention` in NLP. The frontend is designed using HTML and Tailwind CSS, providing a clean and aesthetically pleasing user interface that allows users to input English text and obtain corresponding Chinese translation results. The backend uses a Seq2Seq model constructed with PyTorch for translation, incorporating the `Attention` mechanism. The [model](model.pth) has been trained and saved. The frontend and backend run on the same server, eliminating the need for cross-origin configurations.

## Features
- **Clean and Aesthetic User Interface**: Designed with Tailwind CSS to provide a good user experience.
- **Input English Text**: Users can input English sentences in the input box.
- **Real-time Translation**: After clicking the “Translate” button, the application will call the backend model for translation and display the Chinese result.
- **Error Handling**: If the translation request fails, the corresponding error message will be displayed.

## Project Structure

```
Project/
├── app.py
├── config.py
├── demo.py
├── model.pth
├── Tatoeba.cmn-en.cmn
├── Tatoeba.cmn-en.en
├── templates/
│   └── index.html
├── vocab.pkl
└── requirements.txt
```
- **app.py**: Flask backend application that handles translation requests.
- **config.py**: Configuration file defining parameters for model training and loading.
- **demo.py**: Contains data processing, model definition, training, and translation functionalities.
- **model.pth**: Trained PyTorch model file.
- **vocab.pkl**: Vocabulary file containing source and target language vocabularies.
- **Tatoeba.cmn-en.cmn** and **Tatoeba.cmn-en.en**: English and Chinese corpus files used for model training.
- **templates/index.html**: Frontend page file.
- **requirements.txt**: Project dependency list.

## Running the Project
1. **Clone the Project**
    ```
    git clone 
    ```
2. **Install Dependencies**
   
   Ensure you have Python 3.6 or higher installed. Then, run the following command in the project root directory to install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
   You can proceed to the model training step or skip it and directly move to `Run Backend`.

## Training the Model
The model has already been trained; you can use it directly by referring to [model.pth](model.pth) in the root directory. Alternatively, you can train the model by following these steps:

1. **Prepare the Corpus**
   
   Ensure the [Tatoeba.cmn-en.en](Tatoeba.cmn-en.en) and [Tatoeba.cmn-en.cmn](Tatoeba.cmn-en.cmn) files are located in the project root directory, with each line corresponding to an English and Chinese sentence pair. You can also use your own corpus.

2. **Run the Training Script**
   
   Run the following command in the project root directory to start training the model:
    ```bash
    python demo.py --max_pairs 5000
    ```
   
   This will use the first 5000 sentence pairs for training. After training is complete, the `model.pth` and `vocab.pkl` files will be saved to the project root directory.
   
   **Note**: The training process may take a long time depending on hardware configuration and the amount of data.

## Running the Backend Service
Ensure that the `model.pth` and `vocab.pkl` files are located in the project root directory. Then, run the following command in the project root directory to start the Flask application:
```
python app.py
```
The backend service will start at [http://localhost:5000](http://localhost:5000).

## Using the Frontend Page
1. **Access the Page**
   
   Open your browser and visit [http://localhost:5000](http://localhost:5000) to see the translation page.

2. **Perform Translation**
   
   - Enter English text in the input box.
   - Click the “Translate” button.
   - The translation result will be displayed below.

   **Note**: Due to hardware limitations, the corpus used by the author is relatively small and does not cover everyday language. Additionally, due to limited time, not much time was spent on model training. Therefore, the translation results using the model and vocabulary in this repository may sometimes be unsatisfactory. Thank you for your understanding!

## Detailed Project Explanation

1. **Configuration Parameters (config.py)**
   
   `config.py` defines the parameters required for model training and loading.
   
   | Parameter Name       | Type       | Default Value          | Description                              |
   |----------------------|------------|------------------------|------------------------------------------|
   | `--batch_size`       | `int`      | `64`                   | Number of samples per batch              |
   | `--enc_emb_dim`      | `int`      | `300`                  | Dimension of encoder embeddings          |
   | `--dec_emb_dim`      | `int`      | `300`                  | Dimension of decoder embeddings          |
   | `--hid_dim`          | `int`      | `768`                  | Number of neurons in the hidden layer    |
   | `--num_epochs`       | `int`      | `20`                   | Total number of training epochs          |
   | `--learning_rate`    | `float`    | `0.0005`               | Learning rate for the optimizer          |
   | `--model_path`       | `str`      | `'model.pth'`          | Path to save the model file               |
   | `--vocab_path`       | `str`      | `'vocab.pkl'`          | Path to save the vocabulary file         |
   | `--src_corpus`       | `str`      | `'Tatoeba.cmn-en.en'`  | Path to the source language corpus file  |
   | `--tgt_corpus`       | `str`      | `'Tatoeba.cmn-en.cmn'` | Path to the target language corpus file  |
   | `--max_pairs`        | `int` or `None` | `None`          | Maximum number of sentence pairs for training |

2. **Model Definition and Training (demo.py)**
   
   `demo.py` includes data processing, model definition, training, and translation functionalities.
   
   **Main Features:**
   
   - **Build Vocabulary**: Generates source and target language vocabularies based on the corpus.
   - **Data Preprocessing**: Converts sentences to corresponding vocabulary indices and performs padding.
   - **Model Definition**: Defines Encoder, Decoder, and Seq2Seq models.
   - **Model Training**: Trains the model using the training data and saves the trained model and vocabulary.
   - **Translation Functionality**: Loads the trained model and vocabulary to translate input English text.

3. **Backend Application (app.py)**
   
   `app.py` uses Flask to build the backend service, handling translation requests and returning results.
   
   **Main Features:**
   
   - **Load Model and Vocabulary**
   - **Define Routes**
   - **Homepage**
   - **Translation Interface**

4. **Frontend Page (templates/index.html)**
   
   The frontend page is designed using Tailwind CSS, providing a user-friendly interface.
   
   **Main Features:**
   - **Input Box**: Allows users to input English text.
   - **Translate Button**: Features a breathing animation effect; clicking it sends a translation request.
   - **Translation Result Display**: Shows the translated Chinese text.
   - **Error Message Display**: Displays corresponding error messages if the translation request fails.

## Notes
- **Model and Vocabulary Paths**
   
   Ensure that `model.pth` and `vocab.pkl` files are located in the project root directory, or adjust the `--model_path` and `--vocab_path` parameters in `app.py` and `demo.py` accordingly.

- **Corpus Format**
   
   The `Tatoeba.cmn-en.en` and `Tatoeba.cmn-en.cmn` files should have one sentence per line for English and Chinese pairs, with an equal number of sentences in each.

- **Runtime Environment**
   
   It is recommended to run in a GPU environment that supports CUDA to accelerate training and translation speed; otherwise, the model will run on the CPU, which might be slower.

## Usage Example
1. **Start the Backend Service**
    ```
    python app.py
    ```
2. **Access the Translation Page**
   
   Open your browser and visit [http://localhost:5000](http://localhost:5000).

3. **Perform Translation**
   
   - Enter English text in the input box, for example: "I will try it".
   - Click the “Translate” button.
   - The translation result will be displayed below, e.g., "我會再試。"

    ![img](img/example.png)

## Acknowledgments
- Thanks to all the resources that provided support and inspiration for this project.
- Thanks to [Tatoeba](https://opus.nlpl.eu/Tatoeba-v2023-04-12.php)

## References

 J. Tiedemann, 2012, [*Parallel Data, Tools and Interfaces in OPUS.*](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf) In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)<br/>

## License

MIT License
