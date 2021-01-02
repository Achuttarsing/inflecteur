# inflecteur

inflecteur is a python inflector for french language based on the [DELA dictionary](http://infolingu.univ-mlv.fr)

## Installation :
```console
$ pip install inflecteur
```

## Control tense :
```python
In [1]: inflecteur.inflect_sentence("Elle boit un café et mange un croissant.", tense='Futur')
'Elle boira un café et mangera un croissant .'

In [2]: d.inflect_sentence("Elle boit un café et mange un croissant.", tense='Passé simple')
'Elle but un café et mangea un croissant .'
```

## Control gender :
```python
In [3]: d.inflect_sentence("Elle boit un café et mange un croissant.", gender='m')
'Il boit un café et mange un croissant .'
```

## Control number :
```python
In [4]: d.inflect_sentence("Elle boit un café et mange un croissant.", number='p')
'Elles boivent des cafés et mangent des croissants .'
```
