//! A sheet metal rule can choose its bend relief shape (`bend_relief_shape`: rectangular or
//! round), saved only when it is round. Older designs need no transformation: a rule without it
//! reads as rectangular, the slot every bend relief was before. The last release refuses a design
//! saved with it (the newer format number).
