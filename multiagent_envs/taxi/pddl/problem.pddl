(define (problem taxiProblem)
  (:domain taxi)
  
  (:objects
    garage locationA locationB destination - location
    taxi - location
    passengerAlice passengerBob - location
  )

  (:init
    (at taxi garage)
    (at passengerAlice locationA)
    (at passengerBob locationB)
    (passenger passengerAlice)
    (passenger passengerBob)
    (free)
  )

  (:goal (and
    (at passengerAlice destination)
    (at passengerBob destination)
  ))
)