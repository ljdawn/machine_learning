class Tribble
    constructor: ->
        @isAlive = true
        Tribble.count++
    test = 1
    makeTrouble: -> console.log test
    @count = 0
    @makeTrouble: -> console.log @count


t1 = new Tribble
t2 = new Tribble
t1.test = 6
t1.makeTrouble()
Tribble.makeTrouble()