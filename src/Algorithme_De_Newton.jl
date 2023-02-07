
@doc doc"""
#### Objet
Cette fonction implémente l'algorithme de Newton pour résoudre un problème d'optimisation sans contraintes

#### Syntaxe
```julia
xmin,fmin,flag,nb_iters = Algorithme_de_Newton(f,gradf,hessf,x0,option)
```

#### Entrées :
   - f       : (Function) la fonction à minimiser
   - gradf   : (Function) le gradient de la fonction f
   - hessf   : (Function) la hessienne de la fonction f
   - x0      : (Array{Float,1}) première approximation de la solution cherchée
   - options : (Array{Float,1})
       - max_iter      : le nombre maximal d'iterations
       - Tol_abs       : la tolérence absolue
       - Tol_rel       : la tolérence relative
       - epsilon       : epsilon pour les tests de stagnation

#### Sorties:
   - xmin    : (Array{Float,1}) une approximation de la solution du problème  : ``\min_{x \in \mathbb{R}^{n}} f(x)``
   - fmin    : (Float) ``f(x_{min})``
   - flag    : (Integer) indique le critère sur lequel le programme s'est arrêté (en respectant cet ordre de priorité si plusieurs critères sont vérifiés)
      - 0    : CN1 
      - 1    : stagnation du xk
      - 2    : stagnation du f
      - 3    : nombre maximal d'itération dépassé
   - nb_iters : (Integer) le nombre d'itérations faites par le programme

#### Exemple d'appel
```@example
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin,fmin,flag,nb_iters = Algorithme_De_Newton(f,gradf,hessf,x0,options)
```
"""
function Algorithme_De_Newton(f::Function,gradf::Function,hessf::Function,x0,options)
    "# Si options == [] on prends les paramètres par défaut"
    if options == []
        max_iter = 100
        Tol_abs = sqrt(eps())
        Tol_rel = 1e-15
        epsilon = 1.e-2
    else
        max_iter = options[1]
        Tol_abs = options[2]
        Tol_rel = options[3]
        epsilon = options[4]
    end
    k = 0
    x = x0
    x_suivant = x0
    CN1 = ( norm(gradf(x)) <= max(Tol_rel * norm(gradf(x0)),Tol_abs))
    Stagnation_itere = false
    Stagnation_f = false
    if !CN1
        while (k < max_iter) && !CN1 && !Stagnation_itere && !Stagnation_f
            x = x_suivant
            d = - hessf(x) \ gradf(x) 
            x_suivant = x + d
            k += 1
            CN1 = ( norm(gradf(x_suivant)) <= max(Tol_rel * norm(gradf(x0)),Tol_abs))
            Stagnation_itere = (norm(x_suivant - x) <= epsilon * max(Tol_rel *norm(x), Tol_abs)) 
            Stagnation_f = (abs(f(x_suivant) - f(x)) <= epsilon * max(Tol_rel * abs(f(x)), Tol_abs))
        end
    end 
    
    
    xmin = x_suivant
    fmin = f(x_suivant)
    if CN1
        flag = 0
    elseif Stagnation_itere
        flag = 1
    elseif Stagnation_f
        flag = 2
    else
        flag = 3
    end
    nb_iters = k
    return xmin,fmin,flag,nb_iters
end
