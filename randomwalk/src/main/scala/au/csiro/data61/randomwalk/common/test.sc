import java.util.concurrent.atomic.LongAdder

val a = new LongAdder
Seq.tabulate(5)( _ => {
    a.increment()
    a.longValue()})
Seq.tabulate(5)( _ => {
    a.increment()
    a.longValue()})

