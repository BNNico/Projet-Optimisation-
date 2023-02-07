@doc doc"""

#### Objet

Minimise une fonction de ``\mathbb{R}^{n}`` à valeurs dans ``\mathbb{R}`` en utilisant l'algorithme des régions de confiance. 

La solution approchées des sous-problèmes quadratiques est calculé 
par le pas de Cauchy ou le pas issu de l'algorithme du gradient conjugue tronqué

#### Syntaxe
```julia
xmin, fxmin, flag, nb_iters = Regions_De_Confiance(algo,f,gradf,hessf,x0,option)
```

#### Entrées :

   - algo        : (String) string indicant la méthode à utiliser pour calculer le pas
        - "gct"   : pour l'algorithme du gradient conjugué tronqué
        - "cauchy": pour le pas de Cauchy
   - f           : (Function) la fonction à minimiser
   - gradf       : (Function) le gradient de la fonction f
   - hessf       : (Function) la hessiene de la fonction à minimiser
   - x0          : (Array{Float,1}) point de départ
   - options     : (Array{Float,1})
     - deltaMax       : utile pour les m-à-j de la région de confiance
                      ``R_{k}=\left\{x_{k}+s ;\|s\| \leq \Delta_{k}\right\}``
     - gamma1, gamma2 : ``0 < \gamma_{1} < 1 < \gamma_{2}`` pour les m-à-j de ``R_{k}``
     - eta1, eta2     : ``0 < \eta_{1} < \eta_{2} < 1`` pour les m-à-j de ``R_{k}``
     - delta0         : le rayon de départ de la région de confiance
     - max_iter       : le nombre maximale d'iterations
     - Tol_abs        : la tolérence absolue
     - Tol_rel        : la tolérence relative
     - epsilon       : epsilon pour les tests de stagnation

#### Sorties:

   - xmin    : (Array{Float,1}) une approximation de la solution du problème : 
               ``\min_{x \in \mathbb{R}^{n}} f(x)``
   - fxmin   : (Float) ``f(x_{min})``
   - flag    : (Integer) un entier indiquant le critère sur lequel le programme s'est arrêté (en respectant cet ordre de priorité si plusieurs critères sont vérifiés)
      - 0    : CN1
      - 1    : stagnation du ``x``
      - 2    : stagnation du ``f``
      - 3    : nombre maximal d'itération dépassé
   - nb_iters : (Integer)le nombre d'iteration qu'à fait le programme

#### Exemple d'appel
```julia
algo="gct"
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin, fxmin, flag, nb_iters = Regions_De_Confiance(algo,f,gradf,hessf,x0,options)
```
"""

function Regions_De_Confiance(algo,f::Function,gradf::Function,hessf::Function,x0,options)
    if options == []
        deltaMax = 10
        gamma1 = 0.5
        gamma2 = 2.00
        eta1 = 0.25
        eta2 = 0.75
        delta0 = 2
        max_iter = 1000
        Tol_abs = sqrt(eps())
        Tol_rel = 1e-15
        epsilon = 1.e-2
    else
        deltaMax = options[1]
        gamma1 = options[2]
        gamma2 = options[3]
        eta1 = options[4]
        eta2 = options[5]
        delta0 = options[6]
        max_iter = options[7]
        Tol_abs = options[8]
        Tol_rel = options[9]
        epsilon = options[10]
    end
    n = length(x0)
    nb_iters = 0
    x = x0
    x_suivant = x0
    delta  = delta0
    CN1 = ( norm(gradf(x)) <= max(Tol_rel * norm(gradf(x0)),Tol_abs))
    Stagnation_itere = false
    Stagnation_f = false
    while (nb_iters < max_iter) && !CN1 && !Stagnation_itere && !Stagnation_f
        if algo == "cauchy"
            s, e = Pas_De_Cauchy(gradf(x),hessf(x),delta)
        else 
            s = Gradient_Conjugue_Tronque(gradf(x),hessf(x),[])
        end
        mks = f(x) + gradf(x)'*s + 0.5*s'*hessf(x)*s
        mk0 = f(x)
        p = (f(x) - f(x + s))/(mk0 - mks)
        if p >= eta1 
            x_suivant = x + s
            CN1 = (norm(gradf(x_suivant)) <= max(Tol_rel * norm(gradf(x0)),Tol_abs))
            Stagnation_itere = (norm(x_suivant - x) <= epsilon * max(Tol_rel *norm(x), Tol_abs)) 
            Stagnation_f = (abs(f(x_suivant) - f(x)) <= epsilon * max(Tol_rel * abs(f(x)), Tol_abs))
        end 
        if p >= eta2
            delta = min(deltaMax, gamma2*delta)
        elseif p >= eta1
            delta = delta
        else
            delta = gamma1*delta
        end
        nb_iters += 1     
        x = x_suivant
    end 


    
    xmin = x
    fxmin = f(xmin)
    if CN1
        flag = 0
    elseif Stagnation_itere
        flag = 1
    elseif Stagnation_f
        flag = 2
    else
        flag = 3
    end
    return xmin, fxmin, flag, nb_iters
end
