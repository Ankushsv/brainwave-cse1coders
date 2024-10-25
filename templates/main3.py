from flask import Flask ,redirect,url_for ,render_template,request,session,flash
from datetime import timedelta

app = Flask(__name__)
 
app.secret_key = 'ankushu'


@app.route("/")
def home():
    return render_template('home.html')

# @app.route("/login", methods=['POST','GET'])
# def login():
#     if (request.method=='POST'):
#         user=request.form['name']
#         session['user']=user
        
#         return redirect(url_for('userpager'))
#     #in redirect we use name of fucntion and not page name
#     else:
#       if "user" in session:
#           return redirect(url_for("userpager"))
      
#       return render_template("login.html")
    
# @app.route('/user')
# def userpager():
#      if 'user'in session:
#          user=session['user']
#          return f"Welcome {user}"
#      else:
#          redirect(url_for("login"))
         
# @app.route("/logout")
# def logout():
#     session.pop("user", None)
#     flash("Successfully Logged OUT","info")
#     return redirect(url_for("login"))

# @app.route("/base")
# def base():
#     return render_template('base.html')

if __name__=="__main__":
    app.run(debug=True)
