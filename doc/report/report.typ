#set document(
  title: [A 6-DOF Flight Mechanics Simulation of the Lockheed Martin P-3 Orion],
  author: "Jack Duignan",
  date: auto
)

#set page(
  paper: "a4",
  margin: (x: 1.8cm, y:1.5cm)
)

#set text(
  font: "New Computer Modern",
  size: 10pt
)

#set par(
  justify: true,
  leading: 0.52em
)

#set table(stroke: (_, y) => if y == 0 { (bottom: 1pt, top: 1pt) })
#show table.cell.where(y: 0): set text(weight: "bold")

// use #show: appendix to being appendices
// #outline(target: heading.where(supplement: [Appendix]), title: [Appendix])
#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

#set math.equation(numbering: "(1)")

#show figure.where(
  kind: table
): set figure.caption(position: top)

#align(center, text(18pt)[
  ENCE464 Assignment 2
])

#align(center, text(18pt)[
  Jack Duignan, Isaac Cone, Daniel Hawes Group 13
])

= Architecture Overview - hard - Isaac

big picture

ALU

FPL

Cache

Intruction decodeing etc.

= Multithreading - easy - Daniel

row selection 
memcopy
barrier

= Cache - hard

= Profiling - easy - Jack

 - Python script

 - gprof outputs and how they were used
 
= Compiler Optimisation - easy 

= Individual Topic 1 Jack Duignan - Branch Prediction

= Individual Topic 2 Isaac Cone - GPU

= Individual Topic 3 Daniel Hawes - SIMD

#pagebreak()
#bibliography("bibliography.bib", title: "References", style: "institute-of-electrical-and-electronics-engineers")