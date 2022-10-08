from django.shortcuts import render 

def server(request, resource=None):
    return render(request, "index.html", {"name": resource or "World"})