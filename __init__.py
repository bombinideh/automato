# import re
# from nltk.stem.snowball import PortugueseStemmer
# from automata.fa.dfa import DFA
# import graphviz
# import os

# class SentimentAutomaton:
#     def __init__(self):
#         # Stemmer para Português
#         self.stemmer = PortugueseStemmer()
#         # Em Windows, garante que executável dot seja encontrado
#         os.environ['PATH'] += os.pathsep + r"C:\Program Files\Graphviz\bin"

#         # Palavras-base por categoria
#         det_words = ['o', 'a', 'este', 'essa', 'do', 'da', 'dos', 'das']
#         pron_words = ['eu', 'nós']
#         subs_words = ['aula', 'curso', 'professor', 'conteúdo', 'plataforma']
#         verbs_words = ['ser', 'estar', 'parecer', 'funcionar', 'gostar']
#         adj_pos_words = ['excelente', 'bom', 'dinâmica', 'interessante']
#         adj_neg_words = ['chato', 'confuso', 'ruim', 'péssimo']
#         adj_neu_words = ['ok', 'normal']
#         adv_words = ['muito', 'bastante', 'um pouco']

#         # Stems gerados dinamicamente
#         self.det = set(self.stemmer.stem(w) for w in det_words)
#         self.pron = set(self.stemmer.stem(w) for w in pron_words)
#         self.subs = set(self.stemmer.stem(w) for w in subs_words)
#         self.verbs = set(self.stemmer.stem(w) for w in verbs_words)
#         self.adj_pos = set(self.stemmer.stem(w) for w in adj_pos_words)
#         self.adj_neg = set(self.stemmer.stem(w) for w in adj_neg_words)
#         self.adj_neu = set(self.stemmer.stem(w) for w in adj_neu_words)
#         self.adv = set(self.stemmer.stem(w) for w in adv_words)

#         # Definição do AFD com estado morto para completude
#         states = {'q0', 'q1', 'q2', 'qPos', 'qNeg', 'qNeu'}
#         input_symbols = {'Det', 'Pron', 'Subs', 'Verb', 'AdjPos', 'AdjNeg', 'AdjNeu', 'Adv'}
#         transitions = {
#             'q0': {'Det': 'q1', 'Pron': 'q1'},
#             'q1': {'Subs': 'q2', 'Verb': 'q2'},
#             'q2': {'AdjPos': 'qPos', 'AdjNeg': 'qNeg', 'AdjNeu': 'qNeu', 'Verb': 'q2', 'Adv': 'q2'}
#         }
#         initial_state = 'q0'
#         final_states = {'qPos', 'qNeg', 'qNeu'}

#         # Estado morto para transições faltantes
#         dead_state = 'qDead'
#         states.add(dead_state)

#         # Completa dicionário de transições para todos os estados
#         for state in list(states):
#             transitions.setdefault(state, {})
#         for state in states:
#             for sym in input_symbols:
#                 transitions[state].setdefault(sym, dead_state)

#         # Constrói o DFA
#         self.dfa = DFA(
#             states=states,
#             input_symbols=input_symbols,
#             transitions=transitions,
#             initial_state=initial_state,
#             final_states=final_states
#         )

#     def tokenize_and_stem(self, sentence: str):
#         raw_tokens = re.findall(r"\w+|[^\s\w]", sentence.lower())
#         stems = [self.stemmer.stem(tok) if tok.isalpha() else tok for tok in raw_tokens]
#         return raw_tokens, stems

#     def classify(self, stems):
#         classes = []
#         for stem in stems:
#             if stem in self.det:
#                 classes.append('Det')
#             elif stem in self.pron:
#                 classes.append('Pron')
#             elif stem in self.subs:
#                 classes.append('Subs')
#             elif stem in self.verbs:
#                 classes.append('Verb')
#             elif stem in self.adj_pos:
#                 classes.append('AdjPos')
#             elif stem in self.adj_neg:
#                 classes.append('AdjNeg')
#             elif stem in self.adj_neu:
#                 classes.append('AdjNeu')
#             elif stem in self.adv:
#                 classes.append('Adv')
#             else:
#                 classes.append('OOV')
#         return classes

#     def analyze(self, sentence: str):
#         tokens, stems = self.tokenize_and_stem(sentence)
#         classes = self.classify(stems)
#         state = self.dfa.initial_state

#         for i, cls in enumerate(classes):
#             if cls == 'OOV':
#                 return {'valid': False, 'error': f"Token '{tokens[i]}' não reconhecido (OOV)."}
#             if cls not in self.dfa.transitions[state]:
#                 expected = list(self.dfa.transitions[state].keys())
#                 return {'valid': False, 'error': f"Posição {i}: token '{tokens[i]}' (classe {cls}) não esperado em {state}; esperava {expected}."}
#             state = self.dfa.transitions[state][cls]

#         if state in self.dfa.final_states:
#             mood = 'positivo' if state == 'qPos' else 'negativo' if state == 'qNeg' else 'neutro'
#             return {'valid': True, 'mood': mood}
#         return {'valid': False, 'error': 'Frase terminou sem alcançar estado final de sentimento.'}

#     def draw(self, output_path: str = 'automato'):
#         dot = graphviz.Digraph(format='svg')
#         dot.node('', shape='none')
#         dot.edge('', self.dfa.initial_state)
#         for state in self.dfa.states:
#             shape = 'doublecircle' if state in self.dfa.final_states else 'circle'
#             dot.node(state, shape=shape)
#         for src, paths in self.dfa.transitions.items():
#             for sym, dst in paths.items():
#                 dot.edge(src, dst, label=sym)
#         try:
#             dot.render(output_path)
#         except graphviz.backend.ExecutableNotFound:
#             dot_path = f"{output_path}.dot"
#             with open(dot_path, 'w', encoding='utf-8') as f:
#                 f.write(dot.source)
#             print(f"Graphviz não encontrado; arquivo DOT salvo em '{dot_path}'.")

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
        det_words = ['O', 'o', 'a', 'este', 'essa', 'do', 'da', 'dos', 'das']
        pron_words = ['eu', 'nós']
        subs_words = ['aula', 'curso', 'professor', 'conteúdo', 'plataforma']
        verbs_words = ['ser', 'estar', 'parecer', 'funcionar', 'é']  # verbos estruturais
        pos_verbs_words = ['gostar']  # verbos de sentimento positivo
        adj_pos_words = ['excelente', 'bom', 'dinâmica', 'interessante', 'ótimo', 'boa']
        adj_neg_words = ['chato', 'confuso', 'ruim', 'péssimo']
        adj_neu_words = ['ok', 'normal']
        adv_words = ['muito', 'bastante', 'um pouco']
        # stop_words = ['.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '“', '”'. 'e']

        #Conjunto de stop_words
        self.stop_words = stopwords_pt.union({'.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '“', '”'})

        # Gera stems dinamicamente
        self.det = set(self.stemmer.stem(w) for w in det_words)
        self.pron = set(self.stemmer.stem(w) for w in pron_words)
        self.subs = set(self.stemmer.stem(w) for w in subs_words)
        self.verbs = set(self.stemmer.stem(w) for w in verbs_words)
        self.pos_verbs = set(self.stemmer.stem(w) for w in pos_verbs_words)
        self.adj_pos = set(self.stemmer.stem(w) for w in adj_pos_words)
        self.adj_neg = set(self.stemmer.stem(w) for w in adj_neg_words)
        self.adj_neu = set(self.stemmer.stem(w) for w in adj_neu_words)
        self.adv = set(self.stemmer.stem(w) for w in adv_words)

        # Definição dos estados e símbolos
        states = {'q0', 'q1', 'q2', 'q3', 'qPos', 'qNeg', 'qNeu'}
        input_symbols = {'Det', 'Pron', 'Subs', 'Verb', 'AdjPos', 'AdjNeg', 'AdjNeu', 'Adv'}
        
        # Transições explícitas
        transitions = {
            'q0': {'Det': 'q1', 'Pron': 'q1'},
            'q1': {'Subs': 'q2', 'Verb': 'q2'},
            'q2': {
                'AdjPos': 'qPos', 'AdjNeg': 'qNeg', 'AdjNeu': 'qNeu',
                'Verb': 'q2', 'Adv': 'q2', 'Det': 'q3'
            },
            'q3': {'Subs': 'qPos'}  # VP NP => sentimento positivo
        }
        initial_state = 'q0'
        final_states = {'qPos', 'qNeg', 'qNeu'}

        # Estado morto para completude
        dead_state = 'qDead'
        states.add(dead_state)

        # Completa transições faltantes
        for state in list(states):
            transitions.setdefault(state, {})
        for state in states:
            for sym in input_symbols:
                transitions[state].setdefault(sym, dead_state)

        # Constrói DFA
        self.dfa = DFA(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state=initial_state,
            final_states=final_states
        )

    def tokenize_and_stem(self, sentence: str):
        raw_tokens = re.findall(r"\w+|[^\s\w]", sentence.lower())
        stems = [self.stemmer.stem(tok) if tok.isalpha() else tok for tok in raw_tokens]
        filtered = [
            (tok, stem)
            for tok, stem in zip(raw_tokens, stems)
            if tok not in self.stop_words
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
                classes.append('AdjPos')  # verbo positivo
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
        return classes

    def analyze(self, sentence: str):
        tokens, stems = self.tokenize_and_stem(sentence)
        classes = self.classify(stems)
        state = self.dfa.initial_state
        
        for i, cls in enumerate(classes):
            if cls == 'OOV':
                return {'valid': False, 'error': f"Token '{tokens[i]}' não reconhecido (OOV)."}
            if cls not in self.dfa.transitions[state]:
                expected = list(self.dfa.transitions[state].keys())
                return {'valid': False, 'error': f"Posição {i}: token '{tokens[i]}' (classe {cls}) não esperado em {state}; esperava {expected}."}
            state = self.dfa.transitions[state][cls]

        if state in self.dfa.final_states:
            mood = 'positivo' if state == 'qPos' else 'negativo' if state == 'qNeg' else 'neutro'
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
    resultado = sa.analyze('O curso é péssimo')
    print(resultado)
    sa.draw('meu_automato')