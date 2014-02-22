square = (x) -> x * x

console.log square(5)
console.log (square(5))

hi = -> 'hi there'
console.log hi()
console.log do -> 'hi there with do'
console.log (-> 'hi there without do')()