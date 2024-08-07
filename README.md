### Ames house pricing prediction
### Software And Tools Requirements 

1. [Github Account](https://github.com)  
2. [Heroku Account](http://heroku.com)  
3. [VSCode IDE](https://code.visualstudio.com/)  
4. [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

Create a new Environment:  


`conda create -p venv python==3.8.19 -y`

Activate the Environment:  

`conda activate venv/`  

Procfile  

(It is giving commands to heroku instance that on start what all Commands that needs to be executed as soon as the app starts)  

Commands that are used  basically related to green unicorn  
green unicorn:Purest python http server for wsgi applications it allows to run python applications  concurrently by running multiple processes(When Multiple request hit the web app it distributes it through multiple instances and many more things)  
 
app:app means we are calling app.py file where our application name is app  


heroku will understand that if requiremnet.txt is here then we have to install all these libraries here 