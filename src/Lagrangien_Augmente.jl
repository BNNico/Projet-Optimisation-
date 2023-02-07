@doc doc"""
#### Objet

Résolution des problèmes de minimisation avec une contrainte d'égalité scalaire par l'algorithme du lagrangien augmenté.

#### Syntaxe
```julia
xmin,fxmin,flag,iter,muks,lambdaks = Lagrangien_Augmente(algo,f,gradf,hessf,c,gradc,hessc,x0,options)
```

#### Entrées
  - algo : (String) l'algorithme sans contraintes à utiliser:
    - "newton"  : pour l'algorithme de Newton
    - "cauchy"  : pour le pas de Cauchy
    - "gct"     : pour le gradient conjugué tronqué
  - f : (Function) la fonction à minimiser
  - gradf       : (Function) le gradient de la fonction
  - hessf       : (Function) la hessienne de la fonction
  - c     : (Function) la contrainte [x est dans le domaine des contraintes ssi ``c(x)=0``]
  - gradc : (Function) le gradient de la contrainte
  - hessc : (Function) la hessienne de la contrainte
  - x0 : (Array{Float,1}) la première composante du point de départ du Lagrangien
  - options : (Array{Float,1})
    1. epsilon     : utilisé dans les critères d'arrêt
    2. tol         : la tolérance utilisée dans les critères d'arrêt
    3. itermax     : nombre maximal d'itération dans la boucle principale
    4. lambda0     : la deuxième composante du point de départ du Lagrangien
    5. mu0, tho    : valeurs initiales des variables de l'algorithme

#### Sorties
- xmin : (Array{Float,1}) une approximation de la solution du problème avec contraintes
- fxmin : (Float) ``f(x_{min})``
- flag : (Integer) indicateur du déroulement de l'algorithme
   - 0    : convergence
   - 1    : nombre maximal d'itération atteint
   - (-1) : une erreur s'est produite
- niters : (Integer) nombre d'itérations réalisées
- muks : (Array{Float64,1}) tableau des valeurs prises par mu_k au cours de l'exécution
- lambdaks : (Array{Float64,1}) tableau des valeurs prises par lambda_k au cours de l'exécution

#### Exemple d'appel
```julia
using LinearAlgebra
algo = "gct" # ou newton|gct
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
c(x) =  (x[1]^2) + (x[2]^2) -1.5
gradc(x) = [2*x[1] ;2*x[2]]
hessc(x) = [2 0;0 2]
x0 = [1; 0]
options = []
xmin,fxmin,flag,iter,muks,lambdaks = Lagrangien_Augmente(algo,f,gradf,hessf,c,gradc,hessc,x0,options)
```

#### Tolérances des algorithmes appelés

Pour les tolérances définies dans les algorithmes appelés (Newton et régions de confiance), prendre les tolérances par défaut définies dans ces algorithmes.

"""
function Lagrangien_Augmente(algo,fonc::Function,contrainte::Function,gradfonc::Function,
        hessfonc::Function,grad_contrainte::Function,hess_contrainte::Function,x0,options)

  if options == []
		epsilon = 1e-2
		tol = 1e-5
		itermax = 1000
		lambda0 = 2
		mu0 = 100
		tho = 2.9
	else
		epsilon = options[1]
		tol = options[2]
		itermax = options[3]
		lambda0 = options[4]
		mu0 = options[5]
		tho = options[6]
	end
  nb_iter = 0
  n = length(x0)
  xmin = zeros(n)
  fxmin = 0
  flag = 0
  iter = 0
  muk = mu0
  muks = [mu0]
  lambdak = lambda0
  lambdaks = [lambda0]
  
  x = x0
  beta = 0.9
  nu_chap = 0.1258925
  alpha = 0.1
  eps0 = 1/mu0
  epsk = eps0
  nuk = nu_chap/(mu0)^(alpha)
  function La0(x1)
    return fonc(x1) + lambdak' * contrainte(x1) + muk/2 * norm(contrainte(x1))^2
  end
  Stagnation_itere = false
  Stagnation_f = false
  prise_en_compte_stagnation = true
  CN1 = ( norm(La0(x)) <= eps0) 
  while nb_iter < itermax && !CN1 && !Stagnation_itere && !Stagnation_f
    # Création de la fonction à minimiser
    function La(x1)
      return fonc(x1) + lambdak' * contrainte(x1) + muk/2 * norm(contrainte(x1))^2
    end
    function gradLa(x1)
      return gradfonc(x1) + lambdak' * grad_contrainte(x1) + muk * contrainte(x1) * grad_contrainte(x1)
    end 
    function hessLa(x1)
      return hessfonc(x1) + lambdak' * hess_contrainte(x1) + muk * (grad_contrainte(x1) * grad_contrainte(x1)' + contrainte(x1) * hess_contrainte(x1))
    end
    # Choix de l'algorithme
    if algo == "newton"
      x_suivant,w,l,v = Algorithme_De_Newton(La,gradLa,hessLa,x,[])
    else
      x_suivant,w,l,v = Regions_De_Confiance(algo,La,gradLa,hessLa,x,[])
    end 

    # Mise à jour des variables
    if norm(contrainte(x_suivant)) < nuk
      lambdak = lambdak + muk*contrainte(x_suivant)
      lambdaks = [lambdaks lambdak] 
      epsk = epsk/muk
      nuk = nuk/(muk)^(beta)
    else 
      muk = tho * muk
      muks = [muks muk]
      epsk = eps0/muk
      nuk = nu_chap/(muk)^(alpha)
    end
    CN1 = ( norm(La(x_suivant)) <= epsk ) 
    nb_iter = nb_iter + 1
    if prise_en_compte_stagnation
      Stagnation_itere = (norm(x_suivant - x) <= epsk) 
      Stagnation_f = (abs(fonc(x_suivant) - fonc(x)) <= epsk)
    end
    x = x_suivant
  end
  xmin = x
  fxmin = fonc(x)
  if nb_iter == itermax
    flag = 1
  end      
  return xmin,fxmin,flag, nb_iter, lambdaks, muks
end

