| cenario            | dataset       | tecnicas                                   |   accuracy |   f1_macro |   f1_weighted |   test_loss |   tamanho_treino |   tempo_segundos |
|:-------------------|:--------------|:-------------------------------------------|-----------:|-----------:|--------------:|------------:|-----------------:|-----------------:|
| cifar10_baseline   | cifar10       | []                                         |     0.8125 |     0.8131 |        0.8131 |      0.5853 |            50000 |            151.7 |
| cifar10_rotation   | cifar10       | ['rotation']                               |     0.8178 |     0.8152 |        0.8152 |      0.6166 |           100000 |            245.1 |
| cifar10_flip       | cifar10       | ['flip']                                   |     0.8274 |     0.8252 |        0.8252 |      0.5513 |           100000 |            242   |
| cifar10_brightness | cifar10       | ['brightness']                             |     0.8148 |     0.8131 |        0.8131 |      0.6305 |           100000 |            250.5 |
| cifar10_blur       | cifar10       | ['blur']                                   |     0.7966 |     0.7959 |        0.7959 |      0.6636 |           100000 |            249.1 |
| cifar10_combined   | cifar10       | ['rotation', 'flip', 'brightness', 'blur'] |     0.8203 |     0.8185 |        0.8185 |      0.543  |           100000 |            259   |
| fashion_baseline   | fashion_mnist | []                                         |     0.8985 |     0.9007 |        0.9007 |      0.2998 |            60000 |            139.5 |
| fashion_rotation   | fashion_mnist | ['rotation']                               |     0.9019 |     0.9039 |        0.9039 |      0.3029 |           120000 |            248.4 |
| fashion_flip       | fashion_mnist | ['flip']                                   |     0.91   |     0.9104 |        0.9104 |      0.265  |           120000 |            243   |
| fashion_brightness | fashion_mnist | ['brightness']                             |     0.9003 |     0.9017 |        0.9017 |      0.3217 |           120000 |            243.4 |
| fashion_blur       | fashion_mnist | ['blur']                                   |     0.9073 |     0.9084 |        0.9084 |      0.2885 |           120000 |            242.5 |
| fashion_combined   | fashion_mnist | ['rotation', 'flip', 'brightness', 'blur'] |     0.8967 |     0.8986 |        0.8986 |      0.2967 |           120000 |            249.8 |