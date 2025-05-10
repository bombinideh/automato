# main.py
from sentiment_automaton import SentimentAutomaton


def main():
    sa = SentimentAutomaton()

    frase = 'A matéria é péssima, mas a aula é boa e o professor é ótimo'

    sent = sa.analyze(frase)
    if not sent['valid']:
        print('Erro de sentimento:', sent['error'])
    else:
        print(f"Mood: {sent['mood']}, caminho: {sent['path']}")
        sa.draw_matplotlib(path=sent['path'], fname='automato.png')

if __name__ == '__main__':
    main()
