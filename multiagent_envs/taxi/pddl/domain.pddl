(define (domain taxi)
  (:requirements :strips :typing)
  (:types location)

  (:predicates
    (at ?taxi - location)
    (at ?passenger - location)
    (passenger ?passenger)
    (in-taxi ?passenger)
    (free)
  )

  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and (at ?taxi ?from) (free))
    :effect (and (at ?taxi ?to) (not (at ?taxi ?from)))
  )

  (:action pickup
    :parameters (?person - location)
    :precondition (and (at ?taxi ?person) (passenger ?person) (at ?taxi ?person) (free))
    :effect (and (in-taxi ?person) (not (at ?taxi ?person)) (not (free)))
  )

  (:action dropoff
    :parameters (?person - location)
    :precondition (and (in-taxi ?person) (at ?taxi ?person))
    :effect (and (at ?taxi ?person) (not (in-taxi ?person)) (free))
  )
)