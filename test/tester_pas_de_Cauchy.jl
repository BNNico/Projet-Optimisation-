@doc doc"""
Tester l'algorithme de Pas de Cauchy

# Entrées :
   * afficher : (Bool) affichage ou non des résultats de chaque test

# Les cas de test (dans l'ordre)
   * fct 1 : x011,x012
   * fct 2 : x021,x022
"""

function  tester_pas_de_Cauchy(Pas_De_Cauchy::Function)
    tol_erreur = eps()
    
    @testset "Cas test 1 g = 0, H = [7 0 ; 0 2] et delta = 1 " begin
        # definition des variables
        g = [0,0]
        H = [7 0; 0 2] # a = 0, b = 0
        delta = 1
        s, e = Pas_De_Cauchy(g,H,delta)

        @test s == zeros(2)
        @test e == 0
    end 
    @testset "Cas test 2 g = [1 ; 2], H = [7 0 ; 0 2] et delta = 1 " begin
        # definition des variables
        g = [1,2]
        H = [7 0 ; 0 2] # a = 15, b = -5
        delta = 1
        s, e = Pas_De_Cauchy(g,H,delta)

        @test isapprox(s, -1/3*g, atol = tol_erreur) 
        @test e == 1
    end
    @testset "Cas test 3 g = [1 ; 2], H = [7 -9 ; 1  2] et delta = 5 " begin
        # definition des variables
        g = [1;2]
        H = [7 -9 ; 1 2] # a = -1, b = -5
        delta = 5
        s, e = Pas_De_Cauchy(g,H,delta)

        @test isapprox(s, -sqrt(5)*g, atol = tol_erreur)
        @test e == -1
    end

    @testset "Cas test 4 g = [1*10^(-1);0], H = [7 -9 ; 1 2] et delta = 5*10^(-5) " begin
        # definition des variables
        g = [1*10^(-1);0]
        H = [7 -9 ; 1 2] # a = 9, b = -0,01
        delta = 5*10^(-5)
        s, e = Pas_De_Cauchy(g,H,delta)

        @test isapprox(s, -(delta/0.1)*g, atol = tol_erreur)  
        @test e == -1
    end
end