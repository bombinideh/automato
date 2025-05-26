# Autômato

Este repositório implementa autômatos em Python voltados para processamento de linguagem natural em Português e oferece uma interface web para análise de sentimento em aulas online.

## Descrição

* **grammar_automaton.py**: Verifica a estrutura gramatical de frases em Português utilizando o SpaCy.
* **sentiment_automaton.py**: Analisa o sentimento de feedbacks de alunos por meio de um autômato e gera visualmente o caminho percorrido.
* **app.py**: Aplicação Streamlit que integra a análise de sentimento em uma interface interativa.

## Pré-requisitos

* Python 3.8+
* SpaCy e o modelo `pt_core_news_sm`
* Streamlit
* Dependências listadas em `requirements.txt`

## Instalação

1. **Clone o repositório**

   ```bash
   git clone https://github.com/bombinideh/automato.git
   cd automato
   ```
2. **Crie e ative um ambiente virtual** (recomendado)

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```
4. **Instale o modelo do SpaCy**

   ```bash
   python -m spacy download pt_core_news_sm
   ```

## Estrutura do Projeto

```
.
├── .gitignore
├── app.py
├── automato.png           # Exemplo de visualização do autômato de sentimento
├── grammar_automaton.py   # Implementação do autômato gramatical
├── meu_automato/          # Definições personalizadas de autômatos
├── requirements.txt
├── sentiment_automaton.py # Classe SentimentAutomaton para análise de sentimentos
└── README.md
```

## Uso

### 1. Verificação de Gramática

```python
from grammar_automaton import verificar_gramatica

frase = "A aula foi ótima, mas poderia ter mais exemplos práticos."
if verificar_gramatica(frase):
    print("Frase correta!")
else:
    print("Frase incorreta!")
```

### 2. Análise de Sentimento

```python
from sentiment_automaton import SentimentAutomaton

sa = SentimentAutomaton()
resultado = sa.analyze(
    "A aula foi ótima, mas poderia ter mais exemplos práticos."
)
print(f"Sentimento: {resultado['mood']}")
sa.draw_matplotlib(path=resultado['path'], fname='automato.png')
```

### 3. Aplicação Web com Streamlit

```bash
streamlit run app.py
```

* Acesse `http://localhost:8501` no seu navegador.
* Digite um feedback de aluno e clique em **Analisar Feedback**.
* Visualize o sentimento detectado e o autômato correspondente.




