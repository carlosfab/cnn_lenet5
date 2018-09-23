# LeNet-5
### Redes Neurais Convolucionais

Artigo: [Redes Neurais Convolucionais com Python - sigmoidal.ai](http://sigmoidal.ai/redes-neurais-convolucionais-python/)

Código relativo ao artigo do meu blog, mostrando como implementar sua primeira Rede Neural Convolucional (*Convolutional Neural Network* – CNN) em Python, baseando-se na arquitetura neural LeNet-5, com aplicação prática no *dataset* MNIST.

---

Proposta por [LeCun (1998) em seu *paper* *Gradient-Based Learning Applied to Document Recognition*](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), a arquitetura LeNet-5 tem foco no reconhecimento de dígitos, e foi pensada em reconhecer os números de CEPs em correspondências.

![Arquitetura da LeNet-5, uma Rede Neural Convolucional (CNN) para reconhecimento de dígitos. Fonte: LeCun (1998).](http://sigmoidal.ai/wp-content/uploads/2018/07/lenet.png)

A Figura acima é a ilustração original do *paper* de LeCun. Em uma análise rápida, vemos que a imagem passada como *input* não é achatada (*flatten*), mas é passada preservando as suas dimensões. Isso é mandatório, para manter a relação espacial entre seus pixels, poisuma imagem achatada perderia essa informação importante.

A LeNet-5 possui três tipos de *layers*:

* Convolutional Layers (CONV);
* Pooling Layers (POOL);
* Fully-Connected Layers (FC).

![Arquitetura da Rede Neural Convolucional LeNet-5. Fonte: Andrew Ng.](http://sigmoidal.ai/wp-content/uploads/2018/07/Screenshot-2018-07-12-02.21.44-1024x333.png)

Resumidamente, a arquitetura da LeNet-5 é composta por uma sequência com as seguintes camadas:

* CNN é composta por um conjunto de 6 filtros (5×5), stride=1.
* POOL (2×2), stride=2, para reduzir o tamanho espacial das matrizes resultantes.
* CNN (5×5) com 16 filtros e stride=1.
* POOL (2×2), stride=2.
* Os mapas de características são achatados (flatten), formando 400 nós (5x5x16) para a próxima camanda FC.
* FC com 120 nós.
* FC com 84 nós.

Para ver mais detalhes sobre a teoria e implementação, [veja o artigo completo.](http://sigmoidal.ai/redes-neurais-convolucionais-python/)
