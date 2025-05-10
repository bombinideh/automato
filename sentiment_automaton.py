import re
from nltk.stem.snowball import PortugueseStemmer
from automata.fa.dfa import DFA
import matplotlib.pyplot as plt
import networkx as nx
from grammar_automaton import verificar_gramatica

class SentimentAutomaton:
    def __init__(self):
        # Stemmer para Português
        self.stemmer = PortugueseStemmer()

        # Palavras-base por categoria
        det_words = ['o', 'a', 'este', 'essa', 'do', 'da', 'dos', 'das', 'um', 'uma', 'aqueles', 'aquelas']
        pron_words = ['eu', 'nós', 'você', 'eles', 'elas', 'isso', 'aquilo']
        subs_words = ['aula', 'curso', 'professor', 'conteúdo', 'plataforma', 'ensino', 'material', 'disciplina', 'mentoria', 'tarefa', 'atividade', 'comentário', 'feedback', 'exercício']
        verbs_words = ['ser', 'estar', 'parecer', 'funcionar', 'é', 'ensinar', 'aprender', 'explicar', 'desenvolver', 'aplicar', 'praticar']
        pos_verbs_words = ['adorar', 'aprender', 'compreender', 'entender', 'achar', 'gostar', 'apreciar', 'amar', 'curtir', 'valorizar', 'aprovado', 'recomendar', 'melhorar', 'fácil', 'motivador', 'encorajado']
        adj_pos_words = ['excelente', 'bom', 'dinâmica', 'interessante', 'ótimo', 'boa', 'incrível', 'maravilhoso', 'eficiente', 'útil', 'esclarecedor', 'prático', 'completo', 'atrativo', 'engajador', 'conhecedor', 'relevante']
        adj_neg_words = ['chato', 'confuso', 'ruim', 'péssimo', 'difícil', 'desorganizado', 'aburrido', 'sem foco', 'tedioso', 'insuportável', 'incompleto', 'desinteressante', 'pobre']
        adj_neu_words = ['ok', 'normal', 'aceitável', 'adequado', 'suficiente', 'razoável', 'indiferente']
        adv_words = ['muito', 'bastante', 'um pouco', 'extremamente', 'totalmente', 'levemente', 'quase', 'rapidamente', 'lentamente']


        self.stop_words = {';', '?', ',', '.', '!', '?', ':', '-', '(', ')', '[', ']', '{', '}', "'", '"', '“', '”', '’'}

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
        input_symbols = {'Det', 'Pron', 'Subs', 'Verb', 'PosVerb', 'AdjPos', 'AdjNeg', 'AdjNeu', 'Adv'}
        transitions = {
            'q0': {sym: 'q0' for sym in input_symbols},  # por padrão, tudo vai pra q0
            'q1': {sym: 'q1' for sym in input_symbols}   # por padrão, q1 volta pra q0
        }

        # Transições específicas
        transitions['q0']['AdjPos'] = 'q1'
        transitions['q0']['AdjNeg'] = 'q1'
        transitions['q0']['AdjNeu'] = 'q1'
        transitions['q0']['PosVerb'] = 'q1'
        
        transitions['q1']['PosVerb'] = 'q1'
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
            elif stem in self.adj_pos:
                classes.append('AdjPos')  
            elif stem in self.verbs:
                classes.append('Verb')
            elif stem in self.pos_verbs:
                classes.append('PosVerb')
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
        #verifica se a frase é gramaticalmente correta
        if verificar_gramatica(sentence):
            tokens, stems = self.tokenize_and_stem(sentence)
            classes = self.classify(stems)
            state = self.dfa.initial_state

            stack = []
            path = [state]              

            for i, cls in enumerate(classes):
                print(f"Token: {tokens[i]}, Classe: {cls}, Estado: {state}")
                if cls == 'OOV':
                    return {'valid': False,
                            'error': f"Token '{tokens[i]}' não reconhecido (OOV).",
                            'path': path}   # opcional: devolve o trecho percorrido

                # pilha de sentimento
                if cls == 'AdjPos' or cls == 'PosVerb':
                    stack.append('Pos')
                elif cls == 'AdjNeg':
                    if stack and stack[-1] == 'Pos':
                        stack.pop()
                    else:
                        stack.append('Neg')

                # transição do DFA
                if cls not in self.dfa.transitions[state]:
                    expected = list(self.dfa.transitions[state].keys())
                    return {'valid': False,
                            'error': (f"Posição {i}: token '{tokens[i]}' "
                                    f"(classe {cls}) não esperado em {state}; "
                                    f"esperava {expected}."),
                            'path': path}
                state = self.dfa.transitions[state][cls]
                path.append(state)          # ★ registra o novo estado

            if state in self.dfa.final_states:
                pos, neg = stack.count('Pos'), stack.count('Neg')
                mood = 'positivo' if pos > neg else 'negativo' if neg > pos else 'neutro'
                return {'valid': True, 'mood': mood, 'path': path}

            return {'valid': False,
                    'error': 'Frase terminou sem alcançar estado final de sentimento.',
                    'path': path}
        else:
            return {'valid': False,
                    'error': 'Frase não é gramaticalmente correta.',
                    'path': []}


    def draw_matplotlib(self, path=None, fname=None, seed=42, show=True):
        """
        Desenha o autômato com networkx e destaca o caminho percorrido.
        • path  – lista de estados devolvida por analyze()
        • fname – se fornecido, salva PNG/SVG; usa extensão do nome
        • show  – exibe na tela (útil em notebooks)
        """
        G = nx.DiGraph()

        # nós
        for st in self.dfa.states:
            G.add_node(st)

        # arestas com rótulo
        edge_labels = {}
        for src, trans in self.dfa.transitions.items():
            for sym, dst in trans.items():
                G.add_edge(src, dst)
                # vários símbolos podem ir para o mesmo dst; junte‐os
                edge_labels.setdefault((src, dst), []).append(sym)

        # layout consistente
        pos = nx.spring_layout(G, seed=seed)

        plt.figure(figsize=(9, 6))
        # nós
        nx.draw_networkx_nodes(G, pos,
                            nodelist=[st for st in G.nodes if st not in self.dfa.final_states],
                            node_size=1800, node_color='lightblue')
        nx.draw_networkx_nodes(G, pos,
                            nodelist=list(self.dfa.final_states),
                            node_size=1800, node_color='lightgreen')
        # todas as arestas em cinza
        nx.draw_networkx_edges(G, pos, width=1.2, alpha=.4)

        # se houver caminho, destaque‐o
        if path and len(path) > 1:
            highlight = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=highlight,
                                width=2.8, edge_color='red')

        # rótulos nos nós
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        # rótulos nas arestas
        nice_labels = {k: ','.join(v) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nice_labels, font_size=8)

        plt.title('Autômato – caminho percorrido', fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        if fname:
            plt.savefig(fname, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

# if __name__ == '__main__':
#     sa = SentimentAutomaton()
#     frase = 'Ela bonita canta bem'
#     res = sa.analyze(frase)
#     print(res)                      
#     sa.draw_matplotlib(path=res['path'], fname='automato.png')

