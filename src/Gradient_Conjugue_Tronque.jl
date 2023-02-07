@doc doc"""
#### Objet
Cette fonction calcule une solution approchée du problème

```math
\min_{||s||< \Delta}  q(s) = s^{t} g + \frac{1}{2} s^{t}Hs
```

par l'algorithme du gradient conjugué tronqué

#### Syntaxe
```julia
s = Gradient_Conjugue_Tronque(g,H,option)
```

#### Entrées :   
   - g : (Array{Float,1}) un vecteur de ``\mathbb{R}^n``
   - H : (Array{Float,2}) une matrice symétrique de ``\mathbb{R}^{n\times n}``
   - options          : (Array{Float,1})
      - delta    : le rayon de la région de confiance
      - max_iter : le nombre maximal d'iterations
      - tol      : la tolérance pour la condition d'arrêt sur le gradient

#### Sorties:
   - s : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \Delta} q(s)``

#### Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(g,H,options)
    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        delta = 2
        max_iter = 100
        tol = 1e-6
    else
        delta = options[1]
        max_iter = options[2]
        tol = options[3]
    end
    n = length(g)
    s = zeros(n)
    j = 0
    p = -g
    while j < 2*n && norm(g) > tol  
        k = p'*H*p
        if k <= 0
            alpha_1, alpha_2 = resolution(s,p,delta)
            if alpha_2 == "une seule solution"
                sigma = alpha_1
            else
                if q(g,H,s + alpha_1*p) < q(g,H,s + alpha_2*p)
                    sigma = alpha_1
                else
                    sigma = alpha_2
                end
            end
            return s + sigma*p
        end
        alpha = g'*g/k
        if norm(s + alpha*p) > delta
            double_alpha = resolution(s,p,delta)
            if abs(double_alpha[1]) == double_alpha[1]
                sigma = double_alpha[1]
            else
                sigma = double_alpha[2]
            end
            return s + sigma*p
        end
        s = s + alpha*p
        g_prec = g # on sauvegarde gj
        g = g + alpha*H*p
        beta = g'*g/(g_prec'*g_prec)
        p = -g + beta*p
        j = j + 1
    end
    return s
end

function resolution(s,p,delta)
    a = norm(p)^2
    b = 2*(s'*p)
    c = norm(s)^2 - delta^2
    alpha1 = 0
    alpha2 = "une seule solution"
    if a == 0
        if b == 0
            alpha1 = 0
        else
            alpha1 = -c/b
        end
    else
        delta = b^2 - 4*a*c
        if delta < 0
            alpha1 = 0
        else
            alpha1 = (-b - sqrt(delta))/(2*a)
            alpha2 = (-b + sqrt(delta))/(2*a)
        end
    end 
    return alpha1, alpha2  
end

function q(g,H,s)
    return g'*s + 1/2*s'*H*s
end
