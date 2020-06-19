(define (domain q1-truck)
   (:predicates (place ?r)
		(package ?b)
		(truck ?g)
		(truck-at ?r)
		(package-at ?b ?r)
		(inTruck ?g ?o))

   (:action move
       :parameters  (?from ?to)
       :precondition (and  (place ?from) (place ?to) (truck-at ?from))
       :effect (and  (truck-at ?to)
		     (not (truck-at ?from))))



   (:action load
       :parameters (?obj ?place ?free)
       :precondition  (and  (package ?obj) (place ?place) (truck ?free)
			    (package-at ?obj ?place) (truck-at ?place) (not (inTruck ?free?obj)))
       :effect (and (not (package-at ?obj ?place)) 
		    (inTruck ?free?obj)))


   (:action unload
       :parameters  (?obj  ?place ?free)
       :precondition  (and  (package ?obj) (place ?place) (truck ?free) (truck-at ?place) (inTruck ?free?obj))
       :effect (and (package-at ?obj ?place)
		    (not (inTruck ?free?obj)))))