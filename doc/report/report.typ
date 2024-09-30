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

#pagebreak()
= Multithreading - easy - Daniel

row selection 
memcopy
barrier

#pagebreak()
= Cache - hard

#pagebreak()
= Profiling - easy - Jack

Profiling was used throughout all stages of this projects development. This was done to identify which areas of the program where the slowest and how often these slow areas where called. From these results optimisations where made to the code to reduce execution time. When selecting areas of the code to optimise the sections called most often were prioritised as these give a larger performance benefit then optimising slower less frequent functions. To make profiling easier the various components of the code where compartmentalised into functions, while this does add some execution time (due to stack overheads) it allows the profiling tool gprof to provide more granular results. 

Profiling was conducted on both optimised and non-optimised code to gain a wholistic understanding of the programs excution. The non-optimised program was profiled and used in the initial development stage and once the program was at an acceptable level profiling switched to using the optimised code as this was the more effient source. A breakdown of the execution times and call counts for a non-optimised run of the program with a 101 node cube over 300 iterations using 30 threads can be seen in .

#figure(
  caption: [The key specifications of the P-3C Orion #cite(<P3LockheedSpecs:online>) #cite(<NASAP3Orion:techreport>).],
  table(
  columns: (35%, 25%),
  align: (left, center),
  table.header([Parameter], [Value]),
  table.hline(stroke: 1pt),
  [Dry weight], [$35,017 "kg"$],
  [Maxium take-off weight], [$64,410 "Kg"$],
  [Long-Range Cruise Operating Point], [$25,000 "ft"$ at $350 "KTAS"$],
  [Patrol Operating Point], [$15,000 "ft"$ at $203 "KTAS"$],
  [Length], [$35.6 " m"$],
  [Wingspan], [$30.4 " m"$],
  [Maximum Endurance], [$16 "hours"$],
  [Engines], [4 x T56-A-14 Allison],
  table.hline(stroke: 1pt),
)) <tab:P3-Orion-Specs>


- Python script

- gprof outputs and how they were used
 
#pagebreak()
= Compiler Optimisation - easy 

#pagebreak()
= Individual Topic 1 Jack Duignan - Branch Prediction

#pagebreak()
= Individual Topic 2 Isaac Cone - GPU

#pagebreak()
= Individual Topic 3 Daniel Hawes - SIMD

#pagebreak()
#bibliography("bibliography.bib", title: "References", style: "institute-of-electrical-and-electronics-engineers")