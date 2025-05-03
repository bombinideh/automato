import re
from nltk.stem.snowball import PortugueseStemmer
from automata.fa.dfa import DFA
import graphviz
import os
import nltk
nltk.download('stopwords')          
from nltk.corpus import stopwords

stopwords_pt = set(stopwords.words('portuguese'))

class SentimentAutomaton:
    def __init__(self):
        # Stemmer para Português
        self.stemmer = PortugueseStemmer()

        # Palavras-base por categoria
        det_words = ['o', 'a', 'este', 'essa', 'do', 'da', 'dos', 'das']
        pron_words = ['eu', 'nós']
        subs_words = ['aula', 'curso', 'professor', 'conteúdo', 'plataforma']
        verbs_words = ['ser', 'estar', 'parecer', 'funcionar', 'é']
        pos_verbs_words = ['gostar']
        adj_pos_words = ['excelente', 'bom', 'dinâmica', 'interessante', 'ótimo', 'boa']
        adj_neg_words = ['chato', 'confuso', 'ruim', 'péssimo']
        adj_neu_words = ['ok', 'normal']
        adv_words = ['muito', 'bastante', 'um pouco']

        self.stop_words = stopwords_pt.union({'.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '“', '”'})

        # Stemming das palavras
        self.det = set(self.stemmer.stem(w) for w in det_words)
        self.pron = set(self.stemmer.stem(w) for w in pron_words)
        self.subs = set(self.stemmer.stem(w) for w in subs_words)
        self.verbs = set(self.stemmer.stem(w) for w in verbs_words)
        self.pos_verbs = set(self.stemmer.stem(w) for w in pos_verbs_words)
        self.adj_pos = set(self.stemmer.stem(w) for w in adj_pos_words)
        self.adj_neg = set(self.stemmer.stem(w) for w in adj_neg_words)
        self.adj_neu = set(self.stemmer.stem(w) for w in adj_neu_words)
        self.adv = set(self.stemmer.stem(w) for w in adv_words)

        # Estados e símbolos
        states = {'q0', 'q1'}
        input_symbols = {'Det', 'Pron', 'Subs', 'Verb', 'AdjPos', 'AdjNeg', 'AdjNeu', 'Adv'}
        transitions = {
            'q0': {sym: 'q0' for sym in input_symbols},  # por padrão, tudo vai pra q0
            'q1': {sym: 'q0' for sym in input_symbols}   # por padrão, q1 volta pra q0
        }

        # Transições específicas
        transitions['q0']['AdjPos'] = 'q1'
        transitions['q0']['AdjNeg'] = 'q1'
        transitions['q0']['AdjNeu'] = 'q1'

        transitions['q1']['AdjPos'] = 'q1'
        transitions['q1']['AdjNeg'] = 'q1'
        transitions['q1']['AdjNeu'] = 'q1'

        # Construção do DFA
        self.dfa = DFA(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state='q0',
            final_states={'q1'}
        )


    def tokenize_and_stem(self, sentence: str):
        raw_tokens = re.findall(r"\w+|[^\s\w]", sentence.lower()) # Tokeniza a frase e deixa tudo em minúsculas
        stems = [self.stemmer.stem(tok) if tok.isalpha() else tok for tok in raw_tokens] #deixa somente o radical
        filtered = [
            (tok, stem)
            for tok, stem in zip(raw_tokens, stems)
            if tok not in self.stop_words #Retira as stop words
        ]
        if not filtered:
            return [], []

        raw_tokens, stems = zip(*filtered)
        return list(raw_tokens), list(stems)

    def classify(self, stems):
        classes = []
        for stem in stems:
            if stem in self.det:
                classes.append('Det')
            elif stem in self.pron:
                classes.append('Pron')
            elif stem in self.subs:
                classes.append('Subs')
            elif stem in self.pos_verbs:
                classes.append('AdjPos')  
            elif stem in self.verbs:
                classes.append('Verb')
            elif stem in self.adj_pos:
                classes.append('AdjPos')
            elif stem in self.adj_neg:
                classes.append('AdjNeg')
            elif stem in self.adj_neu:
                classes.append('AdjNeu')
            elif stem in self.adv:
                classes.append('Adv')
            else:
                classes.append('OOV')
        print(f"Classes: {classes}")
        return classes

    def analyze(self, sentence: str):
        tokens, stems = self.tokenize_and_stem(sentence)
        classes = self.classify(stems)
        state = self.dfa.initial_state

        stack = []  # <- aqui está a nossa pilha para sentimentos

        for i, cls in enumerate(classes):
            if cls == 'OOV':
                return {'valid': False, 'error': f"Token '{tokens[i]}' não reconhecido (OOV)."}

            # Simula a pilha
            if cls == 'AdjPos':
                stack.append('Pos')
            elif cls == 'AdjNeg':
                if stack and stack[-1] == 'Pos':
                    stack.pop()  # remove sentimento positivo da pilha
                else:
                    stack.append('Neg')

            # Validação com o DFA original
            if cls not in self.dfa.transitions[state]:
                expected = list(self.dfa.transitions[state].keys())
                return {'valid': False, 'error': f"Posição {i}: token '{tokens[i]}' (classe {cls}) não esperado em {state}; esperava {expected}."}
            state = self.dfa.transitions[state][cls]

        if state in self.dfa.final_states:
            # Avalia pilha
            pos = stack.count('Pos')
            neg = stack.count('Neg')
            if pos > neg:
                mood = 'positivo'
            elif neg > pos:
                mood = 'negativo'
            else:
                mood = 'neutro'
            return {'valid': True, 'mood': mood}

        return {'valid': False, 'error': 'Frase terminou sem alcançar estado final de sentimento.'}

    def draw(self, output_path: str = 'automato'):
        dot = graphviz.Digraph(format='svg')
        dot.node('', shape='none')
        dot.edge('', self.dfa.initial_state)
        for state in self.dfa.states:
            shape = 'doublecircle' if state in self.dfa.final_states else 'circle'
            dot.node(state, shape=shape)
        for src, paths in self.dfa.transitions.items():
            for sym, dst in paths.items():
                dot.edge(src, dst, label=sym)
        try:
            dot.render(output_path)
        except graphviz.backend.ExecutableNotFound:
            with open(f"{output_path}.dot", 'w', encoding='utf-8') as f:
                f.write(dot.source)
            print(f"Graphviz não encontrado; arquivo DOT salvo em '{output_path}.dot'.")



# Exemplo de uso:
if __name__ == '__main__':
    sa = SentimentAutomaton()
    resultado = sa.analyze('A plataforma funciona')
    print(resultado)
    sa.draw('meu_automato')